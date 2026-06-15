"""Teacher model for OPD — loads a large LM and computes per-token logprobs.

Two modes are supported:

1. **Local** (``OPDTeacher``) — loads the model in-process with 4-bit quantization.
   Used when the teacher shares GPUs with the student (memory-tight).

2. **Remote** (``OPDTeacherClient``) — connects to a teacher server running on
   dedicated GPU(s).  Recommended to avoid OOM when teacher + student compete
   for memory on the same GPU.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ============================================================================
# Local teacher (loads model in-process)
# ============================================================================

class OPDTeacher:
    """Wrapper around a large teacher model loaded in-process.

    Parameters
    ----------
    model_path : str
        Path to the teacher model (e.g. Qwen3-32B-Base).
    torch_dtype : torch.dtype or str
        Compute dtype for inference (e.g. ``torch.bfloat16``).
    device_map : str, optional
        Device map strategy. Default ``"auto"``.
    load_in_4bit : bool
        Use BitsAndBytes 4-bit (NF4) quantization.  Default True.
    bnb_4bit_compute_dtype : torch.dtype or str
        Compute dtype for 4-bit layers.  Default ``torch.bfloat16``.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype | str = "auto",
        device_map: str = "auto",
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: torch.dtype | str = "bfloat16",
    ):
        self.model_path = model_path
        self.device_map = device_map

        if isinstance(torch_dtype, str) and torch_dtype not in ("auto", None):
            torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        elif torch_dtype in ("auto", None):
            torch_dtype = torch.bfloat16

        if isinstance(bnb_4bit_compute_dtype, str):
            bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.bfloat16)

        model_kwargs: dict = dict(trust_remote_code=True)

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            print(f"[OPDTeacher] Loading teacher with 4-bit (NF4) quantization")
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = device_map

        print(f"[OPDTeacher] Loading teacher from: {model_path}")
        print(f"[OPDTeacher]   dtype={torch_dtype}, device_map={device_map}, "
              f"load_in_4bit={load_in_4bit}")

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[OPDTeacher] Loaded. Parameters: {param_count/1e9:.1f}B")

    @torch.no_grad()
    def compute_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token teacher log-probabilities for each sequence."""
        try:
            device = (self.model.device if hasattr(self.model, "device")
                      else next(self.model.parameters()).device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        except (RuntimeError, StopIteration):
            pass

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        logprobs = F.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs[:, :-1, :].gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        bsz = token_logprobs.shape[0]
        return F.pad(token_logprobs, (1, 0), value=0.0)

    def __repr__(self) -> str:
        return f"OPDTeacher(path={self.model_path!r}, device_map={self.device_map!r})"


# ============================================================================
# Remote teacher client (connects to teacher server on dedicated GPU)
# ============================================================================

class OPDTeacherClient:
    """Thin HTTP client for a remote teacher server.

    Parameters
    ----------
    server_url : str
        Base URL of the teacher server (e.g. ``"http://127.0.0.1:8100"``).
    """

    def __init__(self, server_url: str):
        if not _HAS_REQUESTS:
            raise ImportError("OPDTeacherClient requires 'requests' package")
        self.server_url = server_url.rstrip("/")

        # Health check
        try:
            r = requests.get(f"{self.server_url}/health", timeout=10)
            r.raise_for_status()
            print(f"[OPDTeacherClient] Connected to teacher server at {self.server_url}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach teacher server at {self.server_url}: {e}"
            ) from e

    def compute_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token teacher logprobs via remote server."""
        payload = {
            "input_ids": input_ids.cpu().tolist(),
            "attention_mask": attention_mask.cpu().tolist(),
        }
        r = requests.post(
            f"{self.server_url}/compute_logprobs",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        logprobs_list = r.json()["logprobs"]
        return torch.tensor(logprobs_list, device=input_ids.device, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"OPDTeacherClient(server_url={self.server_url!r})"

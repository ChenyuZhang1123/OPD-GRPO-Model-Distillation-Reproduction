"""OPD Trainer — HuggingFace Trainer subclass for On-Policy Distillation.

Handles the custom OPD training loop: generate from student → teacher logprobs →
reverse-KL advantages → importance-sampled policy gradient.

This integrates with DeepSpeed, mixed precision, checkpointing, and logging
through the standard HuggingFace Trainer infrastructure (same as GRPO).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_utils import has_length

from src.opd.teacher import OPDTeacher, OPDTeacherClient


class OPDTrainer(Trainer):
    """Custom Trainer for On-Policy Distillation.

    Overrides ``compute_loss`` to inject OPD-specific logic:
    generation from the student, teacher logprob scoring, and reverse-KL loss.

    Parameters
    ----------
    teacher : OPDTeacher or OPDTeacherClient
        Teacher for per-token logprob computation (local or remote).
    tokenizer :
        Tokenizer (used for generation).
    generation_kwargs : dict
        Keyword arguments for ``model.generate()``.
    """

    def __init__(
        self,
        teacher: OPDTeacher | OPDTeacherClient,
        tokenizer,
        generation_kwargs: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.opd_teacher = teacher
        self.opd_tokenizer = tokenizer
        self.opd_gen_kwargs = generation_kwargs or {}

    # ------------------------------------------------------------------
    # compute_loss — the heart of OPD
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """OPD training step.

        *inputs* contains raw dataset fields (including "prompt").  We:
        1. Tokenize prompts
        2. Generate completions from the student (no grad)
        3. Run student forward (with grad) to get new-logprobs
        4. old_logprobs = new_logprobs.detach() (no separate forward needed)
        5. Compute teacher logprobs (no grad)
        6. Compute clipped importance-sampling loss with reverse-KL advantages
        """
        prompts = inputs.pop("prompt")  # list[str]
        prompt_answer = inputs.pop("answer", None)  # not used in OPD loss

        # ---- 1. Tokenize prompts ----
        prompt_enc = self.opd_tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.opd_gen_kwargs.get("max_prompt_length", 2048),
        )

        # Determine device from model (handles DeepSpeed engine)
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda:0")

        prompt_ids = prompt_enc.input_ids.to(device)
        prompt_mask = prompt_enc.attention_mask.to(device)
        prompt_len = prompt_ids.shape[1]

        # ---- 2. Generate completions from student (no grad) ----
        unwrapped = model.module if hasattr(model, "module") else model
        was_training = unwrapped.training
        unwrapped.eval()
        # Gradient checkpointing disables KV cache → O(L²) generation cost.
        # Temporarily re-enable it for fast autoregressive sampling.
        cache_was_enabled = unwrapped.config.use_cache
        unwrapped.config.use_cache = True

        num_gen = self.opd_gen_kwargs.get("num_generations", 1)

        with torch.no_grad():
            generated = unwrapped.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.opd_gen_kwargs.get("max_new_tokens", 1792),
                temperature=self.opd_gen_kwargs.get("temperature", 1.0),
                do_sample=self.opd_gen_kwargs.get("temperature", 1.0) > 0,
                top_p=self.opd_gen_kwargs.get("top_p", 1.0),
                num_return_sequences=num_gen,
                pad_token_id=self.opd_tokenizer.pad_token_id,
                eos_token_id=self.opd_tokenizer.eos_token_id,
            )

        unwrapped.config.use_cache = cache_was_enabled
        if was_training:
            unwrapped.train()

        full_ids = generated  # (B, L_prompt + L_completion)
        full_mask = (full_ids != self.opd_tokenizer.pad_token_id).long()

        # ---- 3. Student new-logprobs (WITH grad, single forward) ----
        # Use the wrapped model so gradients flow through DeepSpeed / LoRA.
        # We compute this FIRST, then detach for old_logprobs — this saves an
        # entire forward pass vs the original 3-pass approach.
        model.train()
        new_logprobs = self._compute_logprobs(model, full_ids, full_mask)

        # ---- 4. old_logprobs = new_logprobs.detach() (no extra forward) ----
        old_logprobs = new_logprobs.detach()

        # ---- 5. Teacher logprobs (no grad) ----
        teacher_logprobs = self.opd_teacher.compute_logprobs(full_ids, full_mask)

        # ---- 6. OPD loss ----
        clip_eps = self.opd_gen_kwargs.get("clip_epsilon", 0.2)
        loss = self._opd_loss(
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            teacher_logprobs=teacher_logprobs,
            attention_mask=full_mask,
            prompt_len=prompt_len,
            clip_epsilon=clip_eps,
        )
        # DeepSpeed requires a strict 0-d scalar
        loss = loss.view(())

        # Store latest metrics for logging
        self._last_opd_metrics = {
            "opd/loss": loss.item(),
            "opd/reverse_kl": (old_logprobs[:, prompt_len:] - teacher_logprobs[:, prompt_len:])
            .mean().item(),
        }

        return (loss, {"loss": loss}) if return_outputs else loss

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, logs, start_time=None):
        if hasattr(self, "_last_opd_metrics"):
            logs.update(self._last_opd_metrics)
        super().log(logs, start_time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_logprobs(self, model, input_ids, attention_mask):
        """Compute per-token logprobs for a sequence."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        logprobs = F.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs[:, :-1, :].gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        # Pad first position with 0
        bsz = token_logprobs.shape[0]
        return F.pad(token_logprobs, (1, 0), value=0.0)

    @staticmethod
    def _opd_loss(old_logprobs, new_logprobs, teacher_logprobs, attention_mask, prompt_len,
                  clip_epsilon: float = 0.2):
        """Clipped importance-sampling loss with reverse-KL advantages.

        advantage[t] = teacher_logprob[t] - old_logprob[t]
        """
        advantages = teacher_logprobs - old_logprobs  # (B, L)

        ratio = torch.exp(new_logprobs - old_logprobs)  # (B, L)

        loss_unclipped = -ratio * advantages
        loss_clipped = -torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        loss_per_token = torch.max(loss_unclipped, loss_clipped)

        # Mask: only completion tokens, skip pad and prompt
        mask = attention_mask.clone()
        mask[:, :prompt_len] = 0  # ignore prompt tokens
        mask[:, 0] = 0             # first position has no logprob

        return (loss_per_token * mask).sum() / mask.sum().clamp(min=1)

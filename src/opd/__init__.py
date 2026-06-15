"""OPD (On-Policy Distillation) training package.

Modularised following the open-r1 architecture, matching the style of ``src/grpo/``.

Modules
-------
configs  : OPDScriptArguments (extends GRPOScriptArguments), YAML loading
teacher  : OPDTeacher — loads a large teacher model and computes per-token logprobs
opd      : Main training entry point with custom OPD training loop
utils    : Data loading, model/tokenizer loading, helpers, logging (mirrors grpo.utils)
"""

from .configs import OPDScriptArguments, load_yaml_config
from .teacher import OPDTeacher
from .utils import get_dataset, get_model, get_tokenizer

__all__ = [
    "get_dataset",
    "get_model",
    "get_tokenizer",
    "load_yaml_config",
    "OPDScriptArguments",
    "OPDTeacher",
]

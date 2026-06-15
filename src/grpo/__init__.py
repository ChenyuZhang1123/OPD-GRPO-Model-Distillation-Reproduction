"""GRPO training package — modularized following the open-r1 architecture.

Modules
-------
configs  : GRPOScriptArguments (extends trl.ScriptArguments), YAML loading
grpo     : Main training entry point — ``main(script_args, training_args, model_args)``
rewards  : Reward functions and registry
utils    : Data loading, model loading, helpers, logging
"""

from .configs import GRPOScriptArguments, load_yaml_config
from .rewards import get_reward_funcs
from .utils import get_dataset, get_model, get_tokenizer

__all__ = [
    "get_dataset",
    "get_model",
    "get_reward_funcs",
    "get_tokenizer",
    "load_yaml_config",
    "GRPOScriptArguments",
]

"""Reward functions for GRPO training."""

import os
import sys
from typing import Callable, List

from src.eval.answer_extraction import extract_and_match

# ============================================================================
# Helpers
# ============================================================================

def _get_rank() -> int:
    """Return LOCAL_RANK or RANK env var, default 0."""
    for key in ("LOCAL_RANK", "RANK"):
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    return 0


def _completion_to_text(completion) -> str:
    """Normalise a completion to str regardless of TRL's internal format.
    TRL may pass completions as str, list[dict] (chat messages), or dict.
    This always returns a plain str.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if content:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        for key in ("content", "text"):
            if key in completion:
                return str(completion[key])
        return str(completion)
    return str(completion)


def _align_answers_to_completions(answers, prompts, completions) -> List[str]:
    """Return an answer list whose length matches len(completions).

    Handles:
      A) len(answers) == len(completions) → pass through
      B) len(answers) == len(prompts) and len(completions) is a multiple of
         len(prompts) → repeat each answer num_generations times
      C) Other mismatches → truncate or pad with "", warn
      D) answers is None → return [""] * len(completions)
    """
    n_comp = len(completions)
    if answers is None:
        return [""] * n_comp

    n_ans = len(answers)
    n_prompt = len(prompts) if prompts is not None else 0

    if n_ans == n_comp:
        return list(answers)

    if n_prompt > 0 and n_ans == n_prompt and n_comp % n_prompt == 0:
        repeat = n_comp // n_prompt
        aligned = []
        for a in answers:
            aligned.extend([a] * repeat)
        return aligned

    # Mismatch — warn and fix
    if _get_rank() == 0:
        print(f"  [WARN] reward answer mismatch: "
              f"len(answers)={n_ans}, len(prompts)={n_prompt}, "
              f"len(completions)={n_comp}")
    if n_ans > n_comp:
        return list(answers[:n_comp])
    else:
        return list(answers) + [""] * (n_comp - n_ans)


# ============================================================================
# Reward functions
# ============================================================================

def format_reward(prompts, completions, completion_ids=None, **kwargs) -> List[float]:
    """Reward 1.0 if the completion contains \\boxed{...}, else 0.0."""
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        if r'\boxed{' in text or r'\boxed ' in text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    # Safety: ensure output length matches input length
    if len(rewards) != len(completions):
        if _get_rank() == 0:
            print(f"  [ERROR] format_reward: len(rewards)={len(rewards)} != "
                  f"len(completions)={len(completions)}. Returning zeros.")
        return [0.0] * len(completions)
    return rewards


def correctness_reward(prompts, completions, completion_ids=None, **kwargs) -> List[float]:
    """Reward 1.0 if extracted answer matches reference answer, else 0.0.

    The reference answer is read from ``kwargs["answer"]`` and aligned to
    completions via ``_align_answers_to_completions``, which handles both
    per-prompt and per-completion answer lists.
    """
    answers = kwargs.get("answer")
    if answers is None:
        return [0.0] * len(completions)

    # Align answers to completions length
    answers = _align_answers_to_completions(answers, prompts, completions)

    # Now len(answers) == len(completions) — safe to zip
    rewards = []
    for completion, ref_ans in zip(completions, answers):
        text = _completion_to_text(completion)
        _, is_correct, _ = extract_and_match(text, str(ref_ans))
        rewards.append(1.0 if is_correct else 0.0)

    if len(rewards) != len(completions):
        if _get_rank() == 0:
            print(f"  [ERROR] correctness_reward: len(rewards)={len(rewards)} != "
                  f"len(completions)={len(completions)}. Returning zeros.")
        return [0.0] * len(completions)
    return rewards


REWARD_FUNCS_REGISTRY: dict[str, Callable] = {
    "format": format_reward,
    "correctness": correctness_reward,
}

def get_reward_funcs(script_args=None) -> List[Callable]:
    """Return the list of reward functions for GRPO training."""
    return [format_reward, correctness_reward]

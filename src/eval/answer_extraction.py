"""
数学答案抽取与规范化。

核心思路：
1. 优先提取最后一个 \\boxed{...}，取花括号内的内容
2. 如果没有 boxed，尝试匹配 "final answer is ..." / "answer is ..." 等模式
3. 优先使用 math-verify 做等价判定（如已安装），回退到增强后的字符串匹配
"""

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Optional math-verify integration
#   pip install math-verify
# ---------------------------------------------------------------------------
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import verify as _mv_verify, parse as _mv_parse  # noqa: F401

    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Answer extraction
# ============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取 \\boxed{...} 内的答案。
    优先取最后一个 \\boxed{...}。
    """
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # 取最后一个匹配
    last_match = matches[-1]
    start = last_match.end() - 1  # 指向'{'

    # 栈匹配嵌套花括号
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1: i]
        i += 1

    # 未闭合的花括号：返回从 start+1 到末尾的内容
    return text[start + 1:]


def extract_answer_heuristic(text: str) -> Optional[str]:
    """
    当没有 \\boxed{} 时，尝试从文本末尾提取答案。
    匹配以下模式（按优先级）：
    - "final answer is X"
    - "answer is X"
    - "therefore X" / "thus X" / "so X"
    - 最后一个数学表达式 $...$ 或 $$...$$
    - 最后一行纯文本
    """
    # 模式 1: "final answer is ..." / "the answer is ..."
    patterns = [
        r'(?:final|the)\s+answer\s+is\s*:?\s*(.+?)(?:\.|\n|$)',
        r'answer\s*:?\s*(.+?)(?:\.|\n|$)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(.+?)(?:\.|\n|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) > 2:
                return candidate

    # 模式 2: 最后一个 $...$ 或 $$...$$ 中的内容
    dollar_matches = re.findall(r'\$\$?(.+?)\$\$?', text)
    if dollar_matches:
        return dollar_matches[-1].strip()

    # 模式 3: 最后一行非空内容（取最后一个非空行）
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # 如果最后一行太长，只取最后一句
        if len(last_line) > 200:
            sentences = re.split(r'(?<=[.!?])\s+', last_line)
            if sentences:
                last_line = sentences[-1]
        return last_line

    return None


# ============================================================================
# Answer normalization
# ============================================================================

def _strip_latex_wrappers(ans: str) -> str:
    """Remove LaTeX display wrappers around an answer string."""
    ans = ans.strip()
    # $$ ... $$
    if ans.startswith('$$') and ans.endswith('$$'):
        ans = ans[2:-2].strip()
    elif ans.startswith('$') and ans.endswith('$'):
        ans = ans[1:-1].strip()
    # \( ... \)
    if ans.startswith(r'\(') and ans.endswith(r'\)'):
        ans = ans[2:-2].strip()
    # \[ ... \]
    if ans.startswith(r'\[') and ans.endswith(r'\]'):
        ans = ans[2:-2].strip()
    # \boxed{...} (in case extraction didn't strip it fully)
    if ans.startswith(r'\boxed{') and ans.endswith('}'):
        ans = ans[7:-1].strip()
    return ans


def _remove_common_prefixes(ans: str) -> str:
    """Remove natural-language answer prefixes."""
    prefixes = [
        r'(?i)^(?:final\s+)?(?:the\s+)?answer\s+is\s*:?\s*',
        r'(?i)^therefore\s*,?\s*',
        r'(?i)^thus\s*,?\s*',
        r'(?i)^hence\s*,?\s*',
        r'(?i)^so\s*,?\s*',
    ]
    for pat in prefixes:
        ans = re.sub(pat, '', ans).strip()
    return ans


def _common_cleanup(ans: str) -> str:
    """Cleanup operations shared by both normalization paths.
    These should NOT remove backslashes (math-verify needs them for LaTeX).
    """
    ans = ans.strip()

    # Strip outer wrappers
    ans = _strip_latex_wrappers(ans)

    # Remove common prefixes
    ans = _remove_common_prefixes(ans)

    # Remove trailing punctuation
    ans = re.sub(r'[.,;:!]\s*$', '', ans).strip()

    # Remove LaTeX formatting commands (but keep structural LaTeX like \frac, \sqrt)
    ans = ans.replace(r'\displaystyle', '')
    ans = re.sub(r'\\left\s*', '', ans)
    ans = re.sub(r'\\right\s*', '', ans)

    # Remove LaTeX spacing
    ans = ans.replace(r'\,', '').replace(r'\;', '').replace(r'\:', '')
    ans = ans.replace(r'\!', '').replace(r'\ ', ' ')

    # Remove outer \text{...}
    while ans.startswith(r'\text{') and ans.endswith('}'):
        ans = ans[6:-1].strip()

    # Normalize whitespace
    ans = re.sub(r'\s+', ' ', ans).strip()

    # Normalize braces/parentheses spacing
    ans = re.sub(r'\s*([{}()\[\]])\s*', r'\1', ans)

    return ans


def normalize_answer(ans: str, for_math_verify: bool = False) -> str:
    """
    规范化答案字符串。

    参数
    ----
    ans : str
        原始答案字符串
    for_math_verify : bool
        如果为 True，保留 LaTeX 结构（如 \\frac）供 math-verify 解析；
        如果为 False，做更激进的字符串替换用于 fallback 匹配。

    返回
    ----
    str
        规范化后的答案字符串
    """
    if ans is None:
        return ''

    ans = str(ans).strip()

    # Common cleanup (no backslash stripping)
    ans = _common_cleanup(ans)

    if for_math_verify:
        # Convert \% to literal % (math-verify handles it)
        ans = ans.replace(r'\%', '%')
        return ans

    # ---- Fallback-specific: convert LaTeX to plain text ----

    # Convert \% to literal %
    ans = ans.replace(r'\%', '%')

    # Percent to decimal conversion (before backslash removal)
    pct_decimal = _percent_to_decimal(ans)
    if pct_decimal is not None:
        ans = pct_decimal
        return ans

    # Convert \frac{a}{b} → (a)/(b) (before removing backslashes)
    ans = _frac_to_slash(ans)

    # Remove remaining backslashes
    ans = ans.replace('\\', '')

    # Normalize operators
    ans = re.sub(r'\s*([+\-*/=<>^])\s*', r' \1 ', ans).strip()

    # Collapse multiple spaces
    ans = re.sub(r'\s+', ' ', ans).strip()

    return ans


def _frac_to_slash(ans: str) -> str:
    """Convert \\frac{a}{b} to (a)/(b) for string-based comparison.
    Must be called BEFORE backslash removal.
    """
    def _replace_frac(m):
        num = m.group(1)
        den = m.group(2)
        return f'({num})/({den})'

    # Match \frac{num}{den} where num/den may themselves contain balanced braces
    ans = re.sub(
        r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        _replace_frac, ans,
    )
    return ans


def _percent_to_decimal(ans: str) -> Optional[str]:
    """If ans looks like 'X%', return the decimal equivalent as string.
    Returns None if not a simple percentage.
    """
    m = re.match(r'^\s*([+-]?\s*(?:\d+\.?\d*|\.\d+))\s*%\s*$', ans)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(' ', '')) / 100.0
        if val == int(val):
            return str(int(val))
        s = f'{val:.10f}'.rstrip('0').rstrip('.')
        return s
    except ValueError:
        return None


# ============================================================================
# Verification
# ============================================================================

def _verify_math_equivalent(pred: str, gold: str) -> Tuple[bool, bool]:
    """
    使用 math-verify 判断两个答案是否数学等价。

    Uses a two-tier approach:
    1. parse() + verify(parsed, parsed) — mathematical equivalence
       (handles 1/2==0.5, 2.0==2, \\frac{1}{2}==0.5, etc.)
    2. verify(string, string) — LaTeX structural matching
       (handles \\sqrt{2}==\\sqrt{2}, \\frac{\\pi}{2}==\\frac{\\pi}{2}, etc.)

    返回
    ----
    (is_equivalent, used_math_verify)
        used_math_verify 为 True 表示成功调用了 math-verify；
        为 False 表示 math-verify 不可用或解析出错，应回退到字符串匹配。
    """
    if not _MATH_VERIFY_AVAILABLE:
        return False, False

    # Normalize for math-verify (preserves LaTeX structure)
    pred = normalize_answer(pred, for_math_verify=True)
    gold = normalize_answer(gold, for_math_verify=True)

    if not pred or not gold:
        return False, False

    # Convert percentages before math-verify (parse('50%') → 50, not 0.5)
    pct_pred = _percent_to_decimal(pred)
    if pct_pred is not None:
        pred = pct_pred
    pct_gold = _percent_to_decimal(gold)
    if pct_gold is not None:
        gold = pct_gold

    # ---- Tier 1: parse() + verify() for mathematical equivalence ----
    try:
        p_parsed = _mv_parse(pred)
        g_parsed = _mv_parse(gold)
    except Exception:
        p_parsed, g_parsed = [], []

    if p_parsed and g_parsed:
        # Both parsed successfully — use mathematical equivalence check
        try:
            result = _mv_verify(p_parsed, g_parsed)
            return bool(result), True
        except Exception:
            pass

    # ---- Tier 2: verify(string, string) for LaTeX structural matching ----
    try:
        result = _mv_verify(pred, gold)
        return bool(result), True
    except Exception:
        return False, False


def _string_match(pred: str, gold: str) -> bool:
    """Enhanced string matching after normalization."""
    pred_norm = normalize_answer(pred, for_math_verify=False)
    gold_norm = normalize_answer(gold, for_math_verify=False)

    if pred_norm == gold_norm:
        return True

    # Also try comparing as lowercase
    if pred_norm.lower() == gold_norm.lower():
        return True

    # Try percent-to-decimal cross-conversion
    pct_pred = _percent_to_decimal(pred)
    pct_gold = _percent_to_decimal(gold)
    if pct_pred is not None:
        pct_pred_norm = normalize_answer(pct_pred, for_math_verify=False)
        if pct_pred_norm == gold_norm:
            return True
    if pct_gold is not None:
        pct_gold_norm = normalize_answer(pct_gold, for_math_verify=False)
        if pred_norm == pct_gold_norm:
            return True
    if pct_pred is not None and pct_gold is not None:
        pp = normalize_answer(pct_pred, for_math_verify=False)
        pg = normalize_answer(pct_gold, for_math_verify=False)
        if pp == pg:
            return True

    return False


# ============================================================================
# Main entry point
# ============================================================================

def extract_and_match(model_output: str, gold_answer: str) -> Tuple[str, bool, str]:
    """
    从模型输出中抽取答案 → 数学等价判定 → 返回结果。

    判定顺序：
    1. 提取答案（boxed → heuristic）
    2. math-verify 等价判定（如可用）
    3. 回退到增强字符串匹配

    参数
    ----
    model_output : str
        模型生成的完整文本
    gold_answer : str
        数据集中的标准答案

    返回
    ----
    Tuple[str, bool, str]
        (extracted_answer, is_correct, method)
        method 为 'boxed', 'boxed_mv', 'heuristic', 'heuristic_mv', 或 'none'
        _mv 后缀表示 math-verify 判定为正确。
    """
    # Step 1: Extract answer
    pred = extract_boxed_answer(model_output)
    method = 'boxed'

    if pred is None:
        pred = extract_answer_heuristic(model_output)
        method = 'heuristic'

    if pred is None:
        return ('', False, 'none')

    # Step 2: Try math-verify first
    is_equiv, used_mv = _verify_math_equivalent(pred, gold_answer)
    if used_mv and is_equiv:
        return (pred, True, method + '_mv')

    # Step 3: Fallback to enhanced string matching
    is_correct = _string_match(pred, gold_answer)
    return (pred, is_correct, method)


# ============================================================================
# Backward-compatible alias
# ============================================================================

def exact_match(pred_answer: str, gold_answer: str) -> bool:
    """
    抽取答案与标准答案的精确匹配（增强版）。
    优先使用 math-verify，回退到字符串比较。
    """
    is_equiv, _ = _verify_math_equivalent(pred_answer, gold_answer)
    if is_equiv:
        return True
    return _string_match(pred_answer, gold_answer)

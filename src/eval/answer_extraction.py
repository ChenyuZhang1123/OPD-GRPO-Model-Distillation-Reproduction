"""
数学答案抽取与规范化。

核心思路：
1. 优先提取最后一个 \\boxed{...}，取花括号内的内容
2. 如果没有 boxed，尝试匹配 "final answer is ..." / "answer is ..." 等模式
3. 对提取到的答案做规范化处理后再做 exact match
"""

import re
from typing import Optional, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取 \\boxed{...} 内的答案。
    优先取最后一个 \\boxed{...}。

    参数
    ----
    text : str
        模型生成的完整文本

    返回
    ----
    Optional[str]
        提取到的 boxed 内容，未找到则返回 None
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
                return text[start + 1 : i]
        i += 1

    # 未闭合的花括号：返回从 start+1 到末尾的内容
    return text[start + 1 :]


def extract_answer_heuristic(text: str) -> Optional[str]:
    """
    当没有 \\boxed{} 时，尝试从文本末尾提取答案。
    匹配以下模式（按优先级）：
    - "final answer is X"
    - "answer is X"
    - "therefore X" / "thus X" / "so X"
    - 最后一个数学表达式 $...$ 或 $$...$$
    - 最后一行纯文本

    参数
    ----
    text : str
        模型生成的完整文本

    返回
    ----
    Optional[str]
        启发式提取的候选答案
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


def normalize_answer(ans: str) -> str:
    """
    规范化答案字符串，用于 exact match 比较。

    操作：
    - 去掉首尾空白
    - 去掉 $$ 和 $ 包裹
    - 去掉 \\text{...} 外层
    - 去掉 \\displaystyle
    - 去掉 \\left, \\right
    - 替换常见 LaTeX 转义（\\% → %）
    - 去掉末尾句号
    - 压缩连续空白

    参数
    ----
    ans : str
        原始答案字符串

    返回
    ----
    str
        规范化后的答案字符串
    """
    ans = ans.strip()

    # 去掉 $$ 和 $ 包裹
    if ans.startswith('$$') and ans.endswith('$$'):
        ans = ans[2:-2].strip()
    elif ans.startswith('$') and ans.endswith('$'):
        ans = ans[1:-1].strip()

    # 去掉 \displaystyle
    ans = ans.replace('\\displaystyle', '')

    # 去掉 \left, \right
    ans = ans.replace('\\left', '').replace('\\right', '')

    # 去掉末尾句号
    if ans.endswith('.'):
        ans = ans[:-1].strip()

    # 替换常见转义
    ans = ans.replace('\\%', '%')

    # 压缩连续空白
    ans = re.sub(r'\s+', ' ', ans)

    # 去掉多余空格
    # 去掉花括号/括号前后的多余空格
    ans = re.sub(r'\s*([{}()\[\]])\s*', r'\1', ans)

    # Strip LaTeX 空格命令
    ans = ans.replace('\\,', '').replace('\\;', '').replace('\\:', '')
    ans = ans.replace('\\!', '').replace('\\ ', ' ')

    # 去掉外层的 \text{...} 包裹
    while ans.startswith('\\text{') and ans.endswith('}'):
        ans = ans[6:-1]

    return ans.strip()


def exact_match(pred_answer: str, gold_answer: str) -> bool:
    """
    抽取答案与标准答案的精确匹配。

    双方都经过 normalize_answer 处理后做字符串比较。

    参数
    ----
    pred_answer : str
        从模型输出中抽取的答案
    gold_answer : str
        数据集中的标准答案

    返回
    ----
    bool
    """
    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)
    return pred_norm == gold_norm


def extract_and_match(model_output: str, gold_answer: str) -> Tuple[str, bool, str]:
    """
    从模型输出中抽取答案 → 规范化 → 比较。

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
        method 为 'boxed', 'heuristic', 或 'none'
    """
    # Step 1: Try \boxed{} extraction
    pred = extract_boxed_answer(model_output)
    method = 'boxed'

    # Step 2: Fallback to heuristic
    if pred is None:
        pred = extract_answer_heuristic(model_output)
        method = 'heuristic'

    if pred is None:
        return ('', False, 'none')

    # Step 3: Normalize and compare
    is_correct = exact_match(pred, gold_answer)
    return (pred, is_correct, method)

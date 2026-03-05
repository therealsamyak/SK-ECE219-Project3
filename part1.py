import os
import re
import random

import torch
import numpy as np


# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
OUTPUT_DIR = "outputs"

# System prompt candidates
SYSTEM_PROMPT_CANDIDATES = [
    "You are a helpful math tutor. Solve the problem step by step, showing all work. Put your final numerical answer in \\boxed{answer}.",
    "You are an expert at solving math problems. Think through the problem carefully, show your reasoning, and place your final answer in \\boxed{answer}.",
    "Please solve this math problem step by step. Explain your thinking process and put your final answer in the format \\boxed{answer}.",
]

# Chosen system prompt
SYSTEM_PROMPT = SYSTEM_PROMPT_CANDIDATES[0]


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU available. Falling back to CPU.")
    return device


DEVICE = get_device()
print(f"Selected device: {DEVICE}")


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...}, handling nested braces.

    Args:
        text: The text to search for \\boxed{...} patterns.

    Returns:
        The content inside the last \\boxed{...} or None if not found.
    """
    boxed_contents = []
    i = 0

    while i < len(text):
        if text[i : i + 7] == r"\boxed{" and (i == 0 or text[i - 1] != "\\"):
            i += 7
            brace_start = i
            brace_count = 1

            while i < len(text) and brace_count > 0:
                if text[i] == "\\" and i + 1 < len(text):
                    i += 2
                elif text[i] == "{":
                    brace_count += 1
                    i += 1
                elif text[i] == "}":
                    brace_count -= 1
                    i += 1
                else:
                    i += 1

            if brace_count == 0:
                content = text[brace_start : i - 1]
                boxed_contents.append(content)

        i += 1

    return boxed_contents[-1] if boxed_contents else None


def extract_ground_truth(raw_answer: str, gt_format: str) -> str:
    """Extract the ground-truth answer string from the dataset's answer field.

    Args:
        raw_answer: The raw answer string from the dataset.
        gt_format: The format of the ground truth ("hashmarks" or "boxed").

    Returns:
        The extracted and normalized ground truth answer.
    """
    if gt_format == "hashmarks":
        m = re.search(r"####\s*(.+)", raw_answer)
        if m:
            return m.group(1).strip().replace(",", "")
        return raw_answer.strip()
    if gt_format == "boxed":
        boxed = extract_boxed(raw_answer)
        if boxed is not None:
            return boxed.strip()
        return raw_answer.strip()
    return raw_answer.strip()


def extract_model_answer(text: str) -> str | None:
    """Extract the model's answer from the response text using fallback chain.

    Fallback order:
    1. \\boxed{...}
    2. Answer: <number>
    3. Final Answer: <number>
    4. Last number in text

    Args:
        text: The model's response text.

    Returns:
        The extracted answer or None if not found.
    """
    boxed_answer = extract_boxed(text)
    if boxed_answer is not None:
        return boxed_answer

    answer_pattern = r"[Aa]nswer\s*[:=]\s*\$?(-?[\d,]+\.?\d*)"
    m = re.search(answer_pattern, text)
    if m:
        return m.group(1)

    final_answer_pattern = r"[Ff]inal\s*[Aa]nswer\s*[:=]\s*\$?(-?[\d,]+\.?\d*)"
    m = re.search(final_answer_pattern, text)
    if m:
        return m.group(1)

    number_pattern = r"(-?\d[\d,]*\.?\d*)"
    numbers = re.findall(number_pattern, text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer: str | None) -> str:
    """Normalize an answer string for comparison.

    Normalization steps:
    - Strip leading/trailing whitespace
    - Remove commas
    - Remove leading $ sign
    - Strip trailing period

    Args:
        answer: The answer string to normalize.

    Returns:
        The normalized answer string.
    """
    if answer is None:
        return ""

    result = answer.strip()
    result = result.replace(",", "")
    if result.startswith("$"):
        result = result[1:]
    if result.endswith("."):
        result = result[:-1]

    return result.strip()


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth after normalization.

    Args:
        predicted: The predicted answer (can be None).
        ground_truth: The ground truth answer.

    Returns:
        True if the normalized answers match, False otherwise.
    """
    norm_predicted = normalize_answer(predicted) if predicted is not None else ""
    norm_ground_truth = normalize_answer(ground_truth)
    return norm_predicted == norm_ground_truth

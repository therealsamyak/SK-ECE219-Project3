import os
import re
import random

import torch
import numpy as np
from datasets import load_dataset, Dataset


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
        True if normalized answers match, False otherwise.
    """
    norm_predicted = normalize_answer(predicted) if predicted is not None else ""
    norm_ground_truth = normalize_answer(ground_truth)
    return norm_predicted == norm_ground_truth


def load_gsm8k_train(num_samples: int | None = None) -> Dataset:
    """Load GSM8K training dataset.

    Args:
        num_samples: Optional number of samples to take from the beginning.
                     If None, returns full training set.

    Returns:
        Dataset with GSM8K training examples.
    """
    dataset = load_dataset("gsm8k", "main", split="train")

    set_seed(SEED)
    dataset = dataset.shuffle(seed=SEED)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def load_gsm8k_test(num_samples: int = 100) -> list[dict]:
    """Load GSM8K test dataset.

    Args:
        num_samples: Number of samples to take from the beginning (default: 100).

    Returns:
        List of dicts with 'question' and 'answer' fields.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    test_examples = []
    for row in dataset:
        test_examples.append({"question": row["question"], "answer": row["answer"]})

    return test_examples


def format_training_example(question: str, answer: str, tokenizer) -> str:
    """Format a GSM8K example for SFT training.

    Converts GSM8K format (question + answer with #### N) to chat format
    with system prompt, user question, and assistant answer using \boxed{N}.

    Args:
        question: The problem question.
        answer: The GSM8K answer (step-by-step reasoning with #### N ending).
        tokenizer: The tokenizer to apply chat template.

    Returns:
        Formatted string ready for SFT training.
    """
    hashmarks_match = re.search(r"####\s*\d+\s*$", answer)

    if hashmarks_match:
        ground_truth = extract_ground_truth(answer, "hashmarks")
        reasoning = re.sub(r"####\s*\d+\s*$", "", answer).strip()
    else:
        numbers = re.findall(r"(-?\d[\d,]*\.?\d*)", answer)
        ground_truth = numbers[-1] if numbers else answer.strip()

        reasoning = answer
        final_answer_patterns = [
            r",\s*so\s+the\s+answer\s+is\s+\d+\s*$",
            r"\.\s*so\s+the\s+answer\s+is\s+\d+\s*$",
            r"so\s+the\s+answer\s+is\s+\d+\s*$",
            r",\s*the\s+answer\s+is\s+\d+\s*$",
            r"\.\s*the\s+answer\s+is\s+\d+\s*$",
        ]
        for pattern in final_answer_patterns:
            reasoning = re.sub(pattern, ".", reasoning, flags=re.IGNORECASE)
        reasoning = reasoning.strip()

    assistant_answer = f"{reasoning}\\boxed{{{ground_truth}}}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_answer},
    ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return formatted


def format_gsm8k_for_sft(dataset, tokenizer) -> Dataset:
    """Format GSM8K dataset for SFT training.

    Applies format_training_example to each row and returns
    a Dataset with a 'text' column.

    Args:
        dataset: The GSM8K dataset (from load_gsm8k_train).
        tokenizer: The tokenizer to apply chat template.

    Returns:
        Dataset with 'text' column containing formatted examples.
    """
    formatted_texts = [
        format_training_example(row["question"], row["answer"], tokenizer)
        for row in dataset
    ]

    return Dataset.from_dict({"text": formatted_texts})

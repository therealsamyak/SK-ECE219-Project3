"""
Part 1: Foundation utilities for GSM8K math reasoning project.
Contains imports, constants, device detection, answer extraction, and data formatting functions.
"""

# Standard library imports
import gc
import json
import math  # noqa: F401
import os
import random
import re

# Third-party imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Constants ──

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
OUTPUT_DIR = "outputs"

# System prompt for math reasoning
SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, showing all work. "
    "Put your final numerical answer in \\boxed{answer}."
)

# Few-shot examples for prompting
FEW_SHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.\n\n\\boxed{6}",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. Then 2 more arrive. So there are 3 + 2 = 5 cars in the parking lot.\n\n\\boxed{5}",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left.\n\n\\boxed{39}",
    ),
]

# ── Device Detection ──


def get_device() -> str:
    """
    Detect and return the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("WARNING: No GPU found. Using CPU (this will be slow).")
    return "cpu"


# Set device at module level
DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Seed Setting ──


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across torch, random, and numpy.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set seed at module level
set_seed(SEED)

# ── Answer Extraction Functions (Task 2) ──


def extract_boxed(text: str) -> str | None:
    """
    Extract content from the last \\boxed{...} in text, handling nested braces.

    Uses stack-based brace matching to correctly handle nested braces.
    Returns the last TOP-LEVEL boxed content (nested boxed expressions are
    considered part of their parent's content).

    Args:
        text: The text to search for boxed content

    Returns:
        The content of the last boxed expression, or None if not found
    """
    boxed_start = r"\boxed{"
    starts = []
    idx = 0
    while True:
        idx = text.find(boxed_start, idx)
        if idx == -1:
            break
        starts.append(idx)
        idx += 1

    if not starts:
        return None

    matches = []
    for start_idx in starts:
        brace_start = start_idx + len(boxed_start) - 1
        stack = []
        content_start = brace_start + 1

        for i in range(brace_start, len(text)):
            char = text[i]
            if char == "{":
                stack.append(i)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        matches.append((start_idx, i, text[content_start:i]))
                        break

    if not matches:
        return None

    top_level = []
    for i, (start_i, end_i, content_i) in enumerate(matches):
        is_contained = False
        for j, (start_j, end_j, _) in enumerate(matches):
            if i != j and start_j < start_i and end_i < end_j:
                is_contained = True
                break
        if not is_contained:
            top_level.append((start_i, end_i, content_i))

    if not top_level:
        return matches[-1][2]

    return top_level[-1][2]


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Strips whitespace, removes commas, removes leading $, strips trailing period.

    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    result = answer.strip()
    result = result.replace(",", "")
    if result.startswith("$"):
        result = result[1:]
    result = result.rstrip(".")
    return result.strip()


def extract_model_answer(text: str) -> str | None:
    """
    Extract the model's answer from generated text using a fallback chain.

    Fallback order:
    1. Extract from \\boxed{...}
    2. Look for "Answer:" or "answer:" pattern
    3. Look for "Final Answer:" or "final answer:" pattern
    4. Use the last number in the text
    5. Return None

    Args:
        text: The generated text to extract an answer from

    Returns:
        The extracted answer string, or None if no answer found
    """
    # Try boxed format first
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)

    # Try "Answer:" or "answer:" pattern
    match = re.search(r"[Aa]nswer\s*[:=]\s*\$?(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # Try "Final Answer:" or "final answer:" pattern
    match = re.search(r"[Ff]inal\s+[Aa]nswer\s*[:=]\s*\$?(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    # Try last number in text
    numbers = re.findall(r"(-?\d[\d,]*\.?\d*)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """
    Check if a predicted answer matches the ground truth.

    Args:
        predicted: The predicted answer (may be None)
        ground_truth: The ground truth answer

    Returns:
        True if answers match after normalization, False otherwise
    """
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def extract_ground_truth(raw_answer: str, gt_format: str = "hashmarks") -> str:
    """
    Extract the ground-truth answer string from the dataset's answer field.

    Args:
        raw_answer: The raw answer string from the dataset
        gt_format: Format of the ground truth ("hashmarks" or "boxed")

    Returns:
        The extracted ground truth answer
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


# ── Data Formatting and GSM8K Loading (Task 3) ──


def load_gsm8k_train(num_samples: int | None = None) -> Dataset:
    """
    Load GSM8K training dataset.

    Args:
        num_samples: Number of samples to load (None for all)

    Returns:
        Dataset object with GSM8K training data
    """
    dataset = load_dataset("gsm8k", "main", split="train")
    if num_samples is not None:
        dataset = dataset.shuffle(seed=SEED).select(range(num_samples))
    return dataset


def load_gsm8k_test(num_samples: int = 100) -> list[dict]:
    """
    Load GSM8K test dataset.

    Args:
        num_samples: Number of test samples to load

    Returns:
        List of dicts with 'question' and 'answer' fields
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = []
    for i in range(min(num_samples, len(dataset))):
        samples.append(
            {
                "question": dataset[i]["question"],
                "answer": dataset[i]["answer"],
            }
        )
    return samples


def format_training_example(question: str, answer: str, tokenizer) -> str:
    """
    Format a GSM8K training example for SFT.

    Extracts ground truth from GSM8K answer (#### pattern) and replaces it with
    \\boxed{answer} format, then formats as chat template.

    Args:
        question: The math question
        answer: The GSM8K answer (with #### ground truth marker)
        tokenizer: Tokenizer to use for chat template formatting

    Returns:
        Formatted text string for training
    """
    # Extract ground truth number from GSM8K answer
    ground_truth = extract_ground_truth(answer, gt_format="hashmarks")

    # Replace #### N ending with \boxed{N} in the answer text
    # The GSM8K format has the reasoning followed by #### <answer>
    answer_text = re.sub(r"####\s*.+$", f"\\\\boxed{{{ground_truth}}}", answer)

    # Format as chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text},
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted


def format_gsm8k_for_sft(dataset: Dataset, tokenizer) -> Dataset:
    """
    Format entire GSM8K dataset for SFT training.

    Args:
        dataset: GSM8K dataset with 'question' and 'answer' fields
        tokenizer: Tokenizer to use for formatting

    Returns:
        Dataset with 'text' column containing formatted training examples
    """

    def format_example(example):
        return {
            "text": format_training_example(
                example["question"], example["answer"], tokenizer
            )
        }

    return dataset.map(format_example)


# ── Model Loading and Evaluation (Task 4) ──


def load_model(model_name=MODEL_NAME, lora_path=None):
    """
    Load model and tokenizer for evaluation.

    Args:
        model_name: HuggingFace model name
        lora_path: Optional path to LoRA adapter

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # LEFT padding for eval
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    tag = f" + LoRA({lora_path})" if lora_path else " (base)"
    print(f"Loaded: {model_name}{tag}")
    return model, tokenizer


def cleanup(*objects):
    """
    Clean up GPU/CPU memory by deleting objects and clearing cache.

    Args:
        *objects: Objects to delete
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU memory freed. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("MPS memory freed.")
    else:
        print("Memory freed (CPU).")


def build_prompts(tokenizer, questions, system_prompt=SYSTEM_PROMPT):
    """
    Build prompts from questions using chat template.

    Args:
        tokenizer: Tokenizer to use for formatting
        questions: List of question strings
        system_prompt: System prompt to use

    Returns:
        List of formatted prompt strings
    """
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def generate_batch(model, tokenizer, questions, system_prompt=SYSTEM_PROMPT):
    """
    Generate responses for a batch of questions.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        questions: List of question strings
        system_prompt: System prompt to use

    Returns:
        List of response strings
    """
    prompts = build_prompts(tokenizer, questions, system_prompt)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    responses = []
    for i in range(len(questions)):
        new_tokens = out[i][prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def generate_batch_few_shot(
    model, tokenizer, questions, few_shot_examples, system_prompt=SYSTEM_PROMPT
):
    """
    Generate responses for a batch of questions using few-shot prompting.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        questions: List of question strings
        few_shot_examples: List of (question, answer) tuples
        system_prompt: System prompt to use

    Returns:
        List of response strings
    """
    prompts = build_few_shot_prompts(
        tokenizer, questions, few_shot_examples, system_prompt
    )
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    responses = []
    for i in range(len(questions)):
        new_tokens = out[i][prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def evaluate_gsm8k(
    model,
    tokenizer,
    num_samples=100,
    batch_size=16,
    system_prompt=SYSTEM_PROMPT,
    few_shot_examples=None,
):
    """
    Evaluate model on GSM8K test set.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        num_samples: Number of test samples to evaluate
        batch_size: Batch size for evaluation
        system_prompt: System prompt to use
        few_shot_examples: Optional list of (question, answer) tuples for few-shot

    Returns:
        Tuple of (accuracy, records) where records is a list of dicts with evaluation details
    """
    test_data = load_gsm8k_test(num_samples=num_samples)
    records = []
    correct = 0

    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i : i + batch_size]
        questions = [item["question"] for item in batch]

        if few_shot_examples is not None:
            responses = generate_batch_few_shot(
                model, tokenizer, questions, few_shot_examples, system_prompt
            )
        else:
            responses = generate_batch(model, tokenizer, questions, system_prompt)

        for j, (item, response) in enumerate(zip(batch, responses)):
            ground_truth = extract_ground_truth(item["answer"])
            extracted = extract_model_answer(response)
            is_correct = answers_match(extracted, ground_truth)
            if is_correct:
                correct += 1
            records.append(
                {
                    "question": item["question"],
                    "ground_truth": ground_truth,
                    "model_response": response,
                    "extracted_answer": extracted,
                    "correct": is_correct,
                }
            )

    accuracy = correct / len(test_data)
    return accuracy, records


# ── Few-Shot Prompting (Task 5) ──


def build_few_shot_prompts(
    tokenizer, questions, few_shot_examples, system_prompt=SYSTEM_PROMPT
):
    """
    Build prompts with few-shot examples using chat template.

    Args:
        tokenizer: Tokenizer to use for formatting
        questions: List of question strings
        few_shot_examples: List of (question, answer) tuples
        system_prompt: System prompt to use

    Returns:
        List of formatted prompt strings with few-shot examples
    """
    prompts = []
    for q in questions:
        messages = [{"role": "system", "content": system_prompt}]
        for example_q, example_a in few_shot_examples:
            messages.append({"role": "user", "content": example_q})
            messages.append({"role": "assistant", "content": example_a})
        messages.append({"role": "user", "content": q})
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


# ── LoRA Parameter Counting (Task 6) ──


def get_lora_config(r=8, alpha=16, dropout=0.05, target_modules=None):
    """
    Create LoRA configuration for fine-tuning.

    Args:
        r: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
        target_modules: List of target module names (defaults to attention layers)

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dict with 'total', 'trainable', and 'percentage' keys
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = (trainable / total) * 100 if total > 0 else 0
    return {"total": total, "trainable": trainable, "percentage": round(percentage, 4)}


def report_parameter_counts(model_name=MODEL_NAME, lora_config=None):
    """
    Load model and report parameter counts before and after LoRA.

    Args:
        model_name: HuggingFace model name
        lora_config: LoRA configuration (uses defaults if None)

    Returns:
        Dict with 'base_total', 'trainable', and 'percentage' keys
    """
    if lora_config is None:
        lora_config = get_lora_config()
    model, tokenizer = load_model(model_name)
    base_params = count_parameters(model)
    model = get_peft_model(model, lora_config)
    lora_params = count_parameters(model)
    result = {
        "base_total": base_params["total"],
        "trainable": lora_params["trainable"],
        "percentage": lora_params["percentage"],
    }
    print(f"Total parameters: {result['base_total']:,}")
    print(f"Trainable LoRA parameters: {result['trainable']:,}")
    print(f"Percentage: {result['percentage']:.4f}%")
    cleanup(model, tokenizer)
    return result


# ── SFT Training (Task 7) ──


def train_sft(
    num_train_samples: int,
    output_dir: str,
    lora_config=None,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 16,
    lr: float = 2e-4,
    max_seq_len: int = 512,
) -> str:
    """
    Train SFT model on GSM8K dataset.

    Args:
        num_train_samples: Number of training samples to use
        output_dir: Directory to save outputs
        lora_config: LoRA configuration (uses defaults if None)
        epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        lr: Learning rate
        max_seq_len: Maximum sequence length

    Returns:
        Path to the saved adapter
    """
    # Load model for training
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Load tokenizer with RIGHT padding for training
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    model = get_peft_model(model, lora_config or get_lora_config())
    model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Load and format training data
    dataset = load_gsm8k_train(num_train_samples)
    train_data = format_gsm8k_for_sft(dataset, tokenizer)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        max_length=max_seq_len,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    print(f"\nStarting SFT training on {num_train_samples} samples...\n")
    trainer.train()

    # Save adapter
    adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")

    cleanup(model, tokenizer, trainer)
    return adapter_path


# ── Utility Functions ──


def save_json(data, filename):
    """Save data to outputs/{filename} as pretty-printed JSON."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {path}")


def save_plot(fig, filename):
    """Save matplotlib figure to outputs/{filename} and close."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def majority_vote(answers: list[str | None]) -> str | None:
    """Return the most common non-None answer, or None if all are None."""
    from collections import Counter

    filtered = [a for a in answers if a is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    return counts.most_common(1)[0][0]


# ── Question Runner Functions ──


def run_q1(args):
    """Q1: Baseline evaluation of base model."""
    model, tokenizer = load_model()
    acc, records = evaluate_gsm8k(
        model, tokenizer, num_samples=args.num_eval, batch_size=args.batch_size
    )
    save_json(
        {
            "accuracy": acc,
            "num_samples": args.num_eval,
            "model": "base",
            "records": records,
        },
        "q1_baseline_accuracy.json",
    )
    print(f"Baseline accuracy: {acc:.2%}")
    cleanup(model, tokenizer)


def run_q2(args):
    """Q2: Identify failure cases from Q1."""
    q1_path = os.path.join(OUTPUT_DIR, "q1_baseline_accuracy.json")
    if os.path.exists(q1_path):
        with open(q1_path) as f:
            q1_data = json.load(f)
        records = q1_data["records"]
    else:
        print("Q1 results not found, running Q1 first...")
        run_q1(args)
        with open(q1_path) as f:
            q1_data = json.load(f)
        records = q1_data["records"]

    failures = [r for r in records if not r["correct"]]
    save_json(failures, "q2_failure_cases.json")
    print(f"Found {len(failures)} failure cases")


def run_q4(args):
    """Q4: Report parameter counts."""
    result = report_parameter_counts()
    save_json(result, "q4_parameter_counts.json")


def run_q5(args):
    """Q5: SFT with 1k samples."""
    adapter_path = train_sft(1000, "outputs/sft_1k")
    model, tokenizer = load_model(lora_path=adapter_path)
    acc, records = evaluate_gsm8k(
        model, tokenizer, num_samples=args.num_eval, batch_size=args.batch_size
    )
    save_json(
        {
            "accuracy": acc,
            "num_samples": args.num_eval,
            "model": "sft_1k",
            "adapter_path": adapter_path,
            "records": records,
        },
        "q5_sft1k_accuracy.json",
    )
    print(f"SFT-1k accuracy: {acc:.2%}")
    cleanup(model, tokenizer)


def run_q7(args):
    """Q7: Scaling study (0, 1k, 3k samples)."""
    # Load baseline accuracy
    q1_path = os.path.join(OUTPUT_DIR, "q1_baseline_accuracy.json")
    if os.path.exists(q1_path):
        with open(q1_path) as f:
            baseline_acc = json.load(f)["accuracy"]
    else:
        baseline_acc = 0.0

    # Load SFT-1k accuracy
    q5_path = os.path.join(OUTPUT_DIR, "q5_sft1k_accuracy.json")
    if os.path.exists(q5_path):
        with open(q5_path) as f:
            sft1k_acc = json.load(f)["accuracy"]
    else:
        sft1k_acc = 0.0

    # Train SFT-3k
    adapter_path = train_sft(3000, "outputs/sft_3k")
    model, tokenizer = load_model(lora_path=adapter_path)
    sft3k_acc, sft3k_records = evaluate_gsm8k(
        model, tokenizer, num_samples=args.num_eval, batch_size=args.batch_size
    )
    print(f"SFT-3k accuracy: {sft3k_acc:.2%}")

    # Save results
    results = {
        "baseline_accuracy": baseline_acc,
        "sft1k_accuracy": sft1k_acc,
        "sft3k_accuracy": sft3k_acc,
        "scaling_points": [
            {"num_examples": 0, "accuracy": baseline_acc},
            {"num_examples": 1000, "accuracy": sft1k_acc},
            {"num_examples": 3000, "accuracy": sft3k_acc},
        ],
        "records": sft3k_records,
    }
    save_json(results, "q7_scaling_accuracy.json")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x_vals = [0, 1000, 3000]
    y_vals = [baseline_acc, sft1k_acc, sft3k_acc]
    ax.plot(x_vals, y_vals, "o-", markersize=10, linewidth=2)
    ax.set_xlabel("Number of Training Examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("SFT Scaling: Accuracy vs Training Data Size")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "q7_accuracy_plot.png")

    cleanup(model, tokenizer)


def run_q8(args):
    """Q8: Compare base vs SFT on Q2 failures."""
    # Load Q2 failures
    q2_path = os.path.join(OUTPUT_DIR, "q2_failure_cases.json")
    if not os.path.exists(q2_path):
        print("Q2 results not found, running Q2 first...")
        run_q2(args)
    with open(q2_path) as f:
        failures = json.load(f)

    # Take first 3 failures
    sample_failures = failures[:3]
    questions = [f["question"] for f in sample_failures]

    # Load base model and generate
    base_model, base_tokenizer = load_model()
    base_responses = generate_batch(base_model, base_tokenizer, questions)

    # Load best SFT adapter (try 3k first, then 1k)
    sft3k_path = "outputs/sft_3k/final_adapter"
    sft1k_path = "outputs/sft_1k/final_adapter"
    if os.path.exists(sft3k_path):
        adapter_path = sft3k_path
    elif os.path.exists(sft1k_path):
        adapter_path = sft1k_path
    else:
        print("No SFT adapter found, training SFT-1k first...")
        adapter_path = train_sft(1000, "outputs/sft_1k")

    sft_model, sft_tokenizer = load_model(lora_path=adapter_path)
    sft_responses = generate_batch(sft_model, sft_tokenizer, questions)

    # Create comparison
    comparisons = []
    for i, fail in enumerate(sample_failures):
        comparisons.append(
            {
                "question": fail["question"],
                "ground_truth": fail["ground_truth"],
                "base_response": base_responses[i],
                "sft_response": sft_responses[i],
            }
        )

    save_json(comparisons, "q8_sft_vs_base_comparison.json")
    print(f"Saved comparison for {len(comparisons)} questions")

    cleanup(base_model, base_tokenizer)
    cleanup(sft_model, sft_tokenizer)


def run_q9(args):
    """Q9: SFT failure cases."""
    # Try Q7 first (3k), then Q5 (1k)
    q7_path = os.path.join(OUTPUT_DIR, "q7_scaling_accuracy.json")
    q5_path = os.path.join(OUTPUT_DIR, "q5_sft1k_accuracy.json")

    if os.path.exists(q7_path):
        with open(q7_path) as f:
            records = json.load(f)["records"]
    elif os.path.exists(q5_path):
        with open(q5_path) as f:
            records = json.load(f)["records"]
    else:
        print("No SFT results found, run Q5 or Q7 first")
        return

    failures = [r for r in records if not r["correct"]][:5]
    save_json(failures, "q9_sft_failures.json")
    print(f"Found {len(failures)} SFT failure cases")


def run_q10(args):
    """Q10: Few-shot evaluation."""
    # Load base model with few-shot
    base_model, base_tokenizer = load_model()
    base_fewshot_acc, base_fewshot_records = evaluate_gsm8k(
        base_model,
        base_tokenizer,
        num_samples=args.num_eval,
        batch_size=args.batch_size,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )
    print(f"Base few-shot accuracy: {base_fewshot_acc:.2%}")

    # Load SFT-3k with few-shot
    sft3k_path = "outputs/sft_3k/final_adapter"
    if not os.path.exists(sft3k_path):
        print("SFT-3k adapter not found, training first...")
        train_sft(3000, "outputs/sft_3k")

    sft_model, sft_tokenizer = load_model(lora_path=sft3k_path)
    sft_fewshot_acc, sft_fewshot_records = evaluate_gsm8k(
        sft_model,
        sft_tokenizer,
        num_samples=args.num_eval,
        batch_size=args.batch_size,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )
    print(f"SFT-3k few-shot accuracy: {sft_fewshot_acc:.2%}")

    # Load zero-shot baselines
    q1_path = os.path.join(OUTPUT_DIR, "q1_baseline_accuracy.json")
    q7_path = os.path.join(OUTPUT_DIR, "q7_scaling_accuracy.json")

    base_zeroshot_acc = 0.0
    sft_zeroshot_acc = 0.0

    if os.path.exists(q1_path):
        with open(q1_path) as f:
            base_zeroshot_acc = json.load(f)["accuracy"]

    if os.path.exists(q7_path):
        with open(q7_path) as f:
            sft_zeroshot_acc = json.load(f)["sft3k_accuracy"]

    results = {
        "base_zeroshot_accuracy": base_zeroshot_acc,
        "base_fewshot_accuracy": base_fewshot_acc,
        "base_delta": base_fewshot_acc - base_zeroshot_acc,
        "sft_zeroshot_accuracy": sft_zeroshot_acc,
        "sft_fewshot_accuracy": sft_fewshot_acc,
        "sft_delta": sft_fewshot_acc - sft_zeroshot_acc,
        "records": {
            "base_fewshot": base_fewshot_records,
            "sft_fewshot": sft_fewshot_records,
        },
    }
    save_json(results, "q10_fewshot_results.json")

    cleanup(base_model, base_tokenizer)
    cleanup(sft_model, sft_tokenizer)


def run_q13(args):
    """Q13: Open challenge - self-consistency with majority voting."""
    # Load best model (SFT-3k)
    sft3k_path = "outputs/sft_3k/final_adapter"
    if not os.path.exists(sft3k_path):
        print("SFT-3k adapter not found, training first...")
        train_sft(3000, "outputs/sft_3k")

    model, tokenizer = load_model(lora_path=sft3k_path)

    # Load test data
    test_data = load_gsm8k_test(num_samples=args.num_eval)
    records = []
    correct = 0
    n_samples = 5  # Number of samples per question

    for item in tqdm(test_data, desc="Self-consistency eval"):
        question = item["question"]
        ground_truth = extract_ground_truth(item["answer"])

        # Generate N responses with sampling
        prompts = build_prompts(tokenizer, [question])
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        answers = []
        responses = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            responses.append(response)
            answer = extract_model_answer(response)
            answers.append(answer)

        # Majority vote
        final_answer = majority_vote(answers)
        is_correct = answers_match(final_answer, ground_truth)
        if is_correct:
            correct += 1

        records.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "final_answer": final_answer,
                "all_answers": answers,
                "all_responses": responses,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(test_data)

    # Load baseline for comparison
    q7_path = os.path.join(OUTPUT_DIR, "q7_scaling_accuracy.json")
    baseline_acc = 0.0
    if os.path.exists(q7_path):
        with open(q7_path) as f:
            baseline_acc = json.load(f)["sft3k_accuracy"]

    results = {
        "method": "self-consistency with majority voting",
        "n_samples_per_question": n_samples,
        "temperature": 0.7,
        "accuracy": accuracy,
        "baseline_sft3k_accuracy": baseline_acc,
        "improvement": accuracy - baseline_acc,
        "records": records,
    }
    save_json(results, "q13_open_challenge.json")
    print(f"Self-consistency accuracy: {accuracy:.2%}")
    print(f"Improvement over baseline: {(accuracy - baseline_acc):.2%}")

    cleanup(model, tokenizer)


# ── CLI Entry Point ──


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Part 1: GSM8K evaluation and SFT training"
    )
    parser.add_argument(
        "--question",
        type=int,
        choices=[1, 2, 4, 5, 7, 8, 9, 10, 13],
        help="Run specific question",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all questions sequentially"
    )
    parser.add_argument(
        "--adapter-path", type=str, help="Path to pre-trained LoRA adapter"
    )
    parser.add_argument(
        "--num-eval", type=int, default=100, help="Number of eval samples"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Eval batch size")
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if output exists"
    )
    args = parser.parse_args()

    # Map question numbers to runner functions and output files
    runners = {
        1: ("q1_baseline_accuracy.json", run_q1),
        2: ("q2_failure_cases.json", run_q2),
        4: ("q4_parameter_counts.json", run_q4),
        5: ("q5_sft1k_accuracy.json", run_q5),
        7: ("q7_scaling_accuracy.json", run_q7),
        8: ("q8_sft_vs_base_comparison.json", run_q8),
        9: ("q9_sft_failures.json", run_q9),
        10: ("q10_fewshot_results.json", run_q10),
        13: ("q13_open_challenge.json", run_q13),
    }

    if args.question:
        questions = [args.question]
    elif args.all:
        questions = [1, 2, 4, 5, 7, 8, 9, 10, 13]
    else:
        parser.print_help()
        return

    for q in questions:
        outfile, runner = runners[q]
        outpath = os.path.join(OUTPUT_DIR, outfile)
        if os.path.exists(outpath) and not args.force:
            print(f"[Q{q}] Skipping — {outfile} already exists (use --force to rerun)")
            continue
        print(f"\n{'=' * 60}\n[Q{q}] Running...\n{'=' * 60}")
        runner(args)
        print(f"[Q{q}] Done.")


if __name__ == "__main__":
    main()

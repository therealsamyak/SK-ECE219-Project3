import gc
import os
import re
import random
import argparse

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import matplotlib.pyplot as plt


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


# ── Model loading ──


def load_model(model_name: str = MODEL_NAME, lora_path: str | None = None):
    """Load a model and tokenizer for evaluation.

    Args:
        model_name: HuggingFace model name/path.
        lora_path: Optional path to LoRA adapter.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
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
    """Delete model/tokenizer objects and free GPU memory.

    Args:
        *objects: Any number of PyTorch objects to delete.
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleanup complete")


# ── Helper functions for saving outputs ──


def save_json(data, filename: str):
    """Save data to outputs/{filename}, pretty-printed.

    Args:
        data: Data to save (dict, list, etc.).
        filename: Name of the file to save (relative to OUTPUT_DIR).
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    import json

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def save_plot(fig, filename: str):
    """Save matplotlib figure to outputs/{filename} and close it.

    Args:
        fig: matplotlib figure object.
        filename: Name of the file to save (relative to OUTPUT_DIR).
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ── Prompt building & batched generation ──


# Fixed pool of few-shot examples from GSM8K training set
FEW_SHOT_EXAMPLES = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48 clips in April.\nIn May, she sold half as many clips, so she sold 48 / 2 = 24 clips.\nThe total number of clips sold is 48 + 24 = 72.\n\\boxed{72}",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns $12 per hour.\nShe worked for 50 minutes, which is 50/60 = 5/6 of an hour.\nHer earnings are $12 * (5/6) = $10.\n\\boxed{10}",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "The wallet costs $100.\nBetty has only half of the money, so she has $100 / 2 = $50.\nHer parents give her $15, and her grandparents give her twice as much, which is 2 * $15 = $30.\nThe total money Betty now has is $50 + $15 + $30 = $95.\nThe amount she still needs is $100 - $95 = $5.\n\\boxed{5}",
    ),
]


def build_prompts(tokenizer, questions, system_prompt: str = SYSTEM_PROMPT):
    """Build chat-formatted prompt strings for a list of questions.

    Args:
        tokenizer: The tokenizer to apply chat template.
        questions: List of question strings.
        system_prompt: The system prompt to use.

    Returns:
        List of formatted prompt strings.
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


def build_few_shot_prompts(
    tokenizer, questions, few_shot_examples, system_prompt: str = SYSTEM_PROMPT
):
    """Build few-shot chat-formatted prompt strings for a list of questions.

    Prepends k complete (question, answer) pairs as user/assistant turns
    before the actual test question.

    Args:
        tokenizer: The tokenizer to apply chat template.
        questions: List of question strings.
        few_shot_examples: List of (question, answer) tuples for demonstrations.
        system_prompt: The system prompt to use.

    Returns:
        List of formatted prompt strings with few-shot examples.
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


def generate_batch(model, tokenizer, questions, system_prompt: str = SYSTEM_PROMPT):
    """Generate responses for a batch of questions in one forward pass.

    Args:
        model: The model to generate with.
        tokenizer: The tokenizer.
        questions: List of question strings.
        system_prompt: The system prompt to use.

    Returns:
        List of generated response strings.
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


# ── LoRA Parameter Counting ──


def count_parameters(model) -> dict:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model to analyze.

    Returns:
        Dict with keys:
        - "total": total number of parameters
        - "trainable": number of trainable parameters
        - "percentage": percentage of trainable params (float)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = (trainable / total) * 100 if total > 0 else 0.0
    return {"total": total, "trainable": trainable, "percentage": percentage}


def get_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Create LoRA configuration for Qwen model.

    Args:
        r: LoRA rank (default: 8).
        alpha: LoRA alpha parameter (default: 16).
        dropout: LoRA dropout rate (default: 0.05).
        target_modules: List of target module names (default: ["q_proj", "k_proj", "v_proj", "o_proj"]).

    Returns:
        LoraConfig object ready for use with get_peft_model.
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


def report_parameter_counts(model) -> dict:
    """Apply LoRA to a model and report parameter counts.

    Args:
        model: Base model to apply LoRA to.

    Returns:
        Dict with parameter counts before and after LoRA application:
        - "before": dict with total, trainable, percentage (base model)
        - "after": dict with total, trainable, percentage (with LoRA)
    """
    before = count_parameters(model)
    lora_config = get_lora_config()
    model_with_lora = get_peft_model(model, lora_config)
    after = count_parameters(model_with_lora)
    return {"before": before, "after": after}


# ── Evaluation runner ──


def evaluate_gsm8k(
    model,
    tokenizer,
    num_samples: int = 100,
    batch_size: int = 16,
    system_prompt: str = SYSTEM_PROMPT,
    few_shot_examples: list | None = None,
) -> tuple[float, list[dict]]:
    """Zero-shot eval on GSM8K test set.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        num_samples: Number of test samples to evaluate (default: 100).
        batch_size: Batch size for generation (default: 16).
        system_prompt: System prompt to use.
        few_shot_examples: Optional list of few-shot examples.

    Returns:
        Tuple of (accuracy, records) where records contains full evaluation data.
    """
    test_examples = load_gsm8k_test(num_samples=num_samples)

    correct_count = 0
    records = []

    for i in tqdm(range(0, len(test_examples), batch_size), desc="Evaluating"):
        batch = test_examples[i : i + batch_size]
        questions = [item["question"] for item in batch]

        responses = generate_batch(model, tokenizer, questions, system_prompt)

        for q, response, item in zip(questions, responses, batch):
            extracted_answer = extract_model_answer(response)
            ground_truth = extract_ground_truth(item["answer"], "hashmarks")
            correct = answers_match(extracted_answer, ground_truth)

            if correct:
                correct_count += 1

            record = {
                "question": q,
                "ground_truth": ground_truth,
                "model_response": response,
                "extracted_answer": extracted_answer,
                "correct": correct,
            }
            records.append(record)

    accuracy = correct_count / len(test_examples)
    return accuracy, records


# ── SFT Training ──


def train_sft(
    num_train_samples: int,
    output_dir: str,
    lora_config: LoraConfig | None = None,
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_seq_len: int = 1024,
) -> str:
    """Train model with SFT on GSM8K dataset.

    Args:
        num_train_samples: Number of training samples to use.
        output_dir: Directory to save outputs.
        lora_config: Optional LoRA config (uses default if None).
        epochs: Number of training epochs (default: 1).
        batch_size: Per-device batch size (default: 8).
        grad_accum: Gradient accumulation steps (default: 4).
        lr: Learning rate (default: 2e-4).
        max_seq_len: Maximum sequence length (default: 1024).

    Returns:
        Path to saved adapter.
    """
    # Load tokenizer with right padding for training
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Apply LoRA (use default if none provided)
    if lora_config is None:
        lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load & format GSM8K training data
    train_data_raw = load_gsm8k_train(num_samples=num_train_samples)
    train_data = format_gsm8k_for_sft(train_data_raw, tokenizer)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

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

    # Train
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

    # Cleanup
    cleanup(model, tokenizer, trainer)

    return adapter_path


# ── Pipeline runner functions ──


def file_exists(filename: str) -> bool:
    """Check if output file exists."""
    return os.path.exists(os.path.join(OUTPUT_DIR, filename))


def run_q1_baseline_eval(
    num_eval: int = 100, batch_size: int = 16, force: bool = False
):
    """Run Q1: baseline evaluation of base model.

    Saves:
        - q1_accuracy.json: {"accuracy": float}
        - q1_records.json: list of evaluation records
    """
    output_acc = "q1_accuracy.json"
    output_records = "q1_records.json"

    if not force and file_exists(output_acc) and file_exists(output_records):
        print("Q1 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q1: Baseline Evaluation ===")
    model, tokenizer = load_model(MODEL_NAME)
    accuracy, records = evaluate_gsm8k(
        model, tokenizer, num_samples=num_eval, batch_size=batch_size
    )

    save_json({"accuracy": accuracy}, output_acc)
    save_json(records, output_records)

    cleanup(model, tokenizer)
    print(f"Q1 complete. Accuracy: {accuracy:.4f}\n")


def run_q2_extract_failures(force: bool = False):
    """Run Q2: extract failure cases from Q1 results.

    Saves:
        - q2_failures.json: list of 3 failure records with excerpts
    """
    output_file = "q2_failures.json"

    if not force and file_exists(output_file):
        print("Q2 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q2: Extract Failures ===")

    import json

    with open(os.path.join(OUTPUT_DIR, "q1_records.json")) as f:
        records = json.load(f)

    failures = [r for r in records if not r["correct"]]

    selected_failures = failures[:3]

    for i, f in enumerate(selected_failures, 1):
        excerpt = (
            f["model_response"][:300] + "..."
            if len(f["model_response"]) > 300
            else f["model_response"]
        )
        print(f"\nFailure {i}:")
        print(f"  Question: {f['question'][:100]}...")
        print(f"  Extracted: {f['extracted_answer']}")
        print(f"  Ground truth: {f['ground_truth']}")
        print(f"  Excerpt: {excerpt}")

    save_json(selected_failures, output_file)
    print(f"Q2 complete. Saved {len(selected_failures)} failure cases.\n")


def run_q4_param_count(force: bool = False):
    """Run Q4: count parameters before and after LoRA.

    Saves:
        - q4_param_counts.json: {"before": {...}, "after": {...}}
    """
    output_file = "q4_param_counts.json"

    if not force and file_exists(output_file):
        print("Q4 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q4: Parameter Counts ===")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    counts = report_parameter_counts(model)

    print(f"Base model params: {counts['before']['total']:,}")
    print(f"Trainable params: {counts['after']['trainable']:,}")
    print(f"Percentage: {counts['after']['percentage']:.4f}%")

    save_json(counts, output_file)
    cleanup(model)
    print("Q4 complete.\n")


def run_q5_train_1k(num_eval: int = 100, batch_size: int = 16, force: bool = False):
    """Run Q5: train SFT-1k and evaluate.

    Saves:
        - q5_accuracy.json: {"accuracy": float}
        - q5_records.json: list of evaluation records
    """
    output_acc = "q5_accuracy.json"
    output_records = "q5_records.json"

    if not force and file_exists(output_acc) and file_exists(output_records):
        print("Q5 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q5: Train SFT-1k ===")

    output_dir = os.path.join(OUTPUT_DIR, "q5_adapter")
    adapter_path = train_sft(1000, output_dir)

    model, tokenizer = load_model(MODEL_NAME, lora_path=adapter_path)
    accuracy, records = evaluate_gsm8k(
        model, tokenizer, num_samples=num_eval, batch_size=batch_size
    )

    save_json({"accuracy": accuracy}, output_acc)
    save_json(records, output_records)

    cleanup(model, tokenizer)
    print(f"Q5 complete. Accuracy: {accuracy:.4f}\n")


def run_q7_train_3k(num_eval: int = 100, batch_size: int = 16, force: bool = False):
    """Run Q7: train SFT-3k and evaluate.

    Saves:
        - q7_3k_accuracy.json: {"accuracy": float}
        - q7_3k_records.json: list of evaluation records
    """
    output_acc = "q7_3k_accuracy.json"
    output_records = "q7_3k_records.json"

    if not force and file_exists(output_acc) and file_exists(output_records):
        print("Q7 (3k) outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q7: Train SFT-3k ===")

    output_dir = os.path.join(OUTPUT_DIR, "q7_3k_adapter")
    adapter_path = train_sft(3000, output_dir)

    model, tokenizer = load_model(MODEL_NAME, lora_path=adapter_path)
    accuracy, records = evaluate_gsm8k(
        model, tokenizer, num_samples=num_eval, batch_size=batch_size
    )

    save_json({"accuracy": accuracy}, output_acc)
    save_json(records, output_records)

    cleanup(model, tokenizer)
    print(f"Q7 (3k) complete. Accuracy: {accuracy:.4f}\n")


def run_q7_plot_scaling(force: bool = False):
    """Run Q7: collect accuracies and plot scaling.

    Saves:
        - q7_scaling.json: {"training_sizes": [...], "accuracies": [...]}
        - q7_scaling_plot.png: matplotlib plot
    """
    output_data = "q7_scaling.json"
    output_plot = "q7_scaling_plot.png"

    if not force and file_exists(output_data) and file_exists(output_plot):
        print("Q7 scaling outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q7: Plot Scaling ===")

    import json

    with open(os.path.join(OUTPUT_DIR, "q1_accuracy.json")) as f:
        acc_base = json.load(f)["accuracy"]

    with open(os.path.join(OUTPUT_DIR, "q5_accuracy.json")) as f:
        acc_1k = json.load(f)["accuracy"]

    with open(os.path.join(OUTPUT_DIR, "q7_3k_accuracy.json")) as f:
        acc_3k = json.load(f)["accuracy"]

    training_sizes = [0, 1000, 3000]
    accuracies = [acc_base, acc_1k, acc_3k]

    scaling_data = {"training_sizes": training_sizes, "accuracies": accuracies}
    save_json(scaling_data, output_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_sizes, accuracies, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Training Examples", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Training Size (LoRA SFT)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(training_sizes)
    for x, y in zip(training_sizes, accuracies):
        ax.annotate(
            f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    save_plot(fig, output_plot)
    print("Q7 scaling complete.\n")


def run_q8_compare_models(force: bool = False):
    """Run Q8: compare base and SFT-3k on Q2 failures.

    Saves:
        - q8_comparison.json: {"failures": [...]}
    """
    output_file = "q8_comparison.json"

    if not force and file_exists(output_file):
        print("Q8 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q8: Compare Models on Failures ===")

    import json

    with open(os.path.join(OUTPUT_DIR, "q2_failures.json")) as f:
        failures = json.load(f)

    questions = [f["question"] for f in failures]

    model_base, tokenizer_base = load_model(MODEL_NAME)
    responses_base = generate_batch(model_base, tokenizer_base, questions)

    model_sft, tokenizer_sft = load_model(
        MODEL_NAME, lora_path=os.path.join(OUTPUT_DIR, "q7_3k_adapter/final_adapter")
    )
    responses_sft = generate_batch(model_sft, tokenizer_sft, questions)

    comparison = []
    for i, (q, resp_base, resp_sft, fail_record) in enumerate(
        zip(questions, responses_base, responses_sft, failures)
    ):
        comparison.append(
            {
                "question": q,
                "base_response": resp_base,
                "sft_response": resp_sft,
                "base_extracted": fail_record["extracted_answer"],
                "sft_extracted": extract_model_answer(resp_sft),
                "ground_truth": fail_record["ground_truth"],
            }
        )

    save_json({"failures": comparison}, output_file)

    cleanup(model_base, tokenizer_base, model_sft, tokenizer_sft)
    print("Q8 complete.\n")


def run_q9_sft_failures(force: bool = False):
    """Run Q9: find SFT-3k failures.

    Saves:
        - q9_failures.json: list of 2 failure records with full responses
    """
    output_file = "q9_failures.json"

    if not force and file_exists(output_file):
        print("Q9 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q9: SFT-3k Failures ===")

    import json

    with open(os.path.join(OUTPUT_DIR, "q7_3k_records.json")) as f:
        records = json.load(f)

    failures = [r for r in records if not r["correct"]]

    selected_failures = failures[:2]

    for i, f in enumerate(selected_failures, 1):
        print(f"\nFailure {i}:")
        print(f"  Question: {f['question'][:100]}...")
        print(f"  Extracted: {f['extracted_answer']}")
        print(f"  Ground truth: {f['ground_truth']}")

    save_json(selected_failures, output_file)
    print(f"Q9 complete. Saved {len(selected_failures)} failure cases.\n")


def run_q10_fewshot(num_eval: int = 100, batch_size: int = 16, force: bool = False):
    """Run Q10: few-shot evaluation of base and SFT-3k.

    Saves:
        - q10_fewshot_results.json: {"base_no_shot": ..., "base_few_shot": ..., "sft_no_shot": ..., "sft_few_shot": ...}
    """
    output_file = "q10_fewshot_results.json"

    if not force and file_exists(output_file):
        print("Q10 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q10: Few-Shot Evaluation ===")

    import json

    with open(os.path.join(OUTPUT_DIR, "q1_accuracy.json")) as f:
        acc_base_no_shot = json.load(f)["accuracy"]

    with open(os.path.join(OUTPUT_DIR, "q7_3k_accuracy.json")) as f:
        acc_sft_no_shot = json.load(f)["accuracy"]

    model_base, tokenizer_base = load_model(MODEL_NAME)
    acc_base_fewshot, _ = evaluate_gsm8k(
        model_base,
        tokenizer_base,
        num_samples=num_eval,
        batch_size=batch_size,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )

    model_sft, tokenizer_sft = load_model(
        MODEL_NAME, lora_path=os.path.join(OUTPUT_DIR, "q7_3k_adapter/final_adapter")
    )
    acc_sft_fewshot, _ = evaluate_gsm8k(
        model_sft,
        tokenizer_sft,
        num_samples=num_eval,
        batch_size=batch_size,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )

    results = {
        "base_no_shot": acc_base_no_shot,
        "base_few_shot": acc_base_fewshot,
        "sft_no_shot": acc_sft_no_shot,
        "sft_few_shot": acc_sft_fewshot,
        "base_delta": acc_base_fewshot - acc_base_no_shot,
        "sft_delta": acc_sft_fewshot - acc_sft_no_shot,
    }

    save_json(results, output_file)

    cleanup(model_base, tokenizer_base, model_sft, tokenizer_sft)
    print("Q10 complete.\n")
    print(
        f"  Base: {acc_base_no_shot:.4f} → {acc_base_fewshot:.4f} (Δ: {results['base_delta']:+.4f})"
    )
    print(
        f"  SFT:  {acc_sft_no_shot:.4f} → {acc_sft_fewshot:.4f} (Δ: {results['sft_delta']:+.4f})\n"
    )


def run_q13_open_challenge(
    num_eval: int = 100, batch_size: int = 16, num_samples: int = 5, force: bool = False
):
    """Run Q13: self-consistency voting.

    Saves:
        - q13_results.json: {"accuracy": float, "num_samples": int, "method": "self_consistency"}
    """
    output_file = "q13_results.json"

    if not force and file_exists(output_file):
        print("Q13 outputs exist, skipping (use --force to rerun)")
        return

    print("\n=== Q13: Self-Consistency Voting ===")

    model, tokenizer = load_model(
        MODEL_NAME, lora_path=os.path.join(OUTPUT_DIR, "q7_3k_adapter/final_adapter")
    )
    test_examples = load_gsm8k_test(num_samples=num_eval)

    correct_count = 0

    for i in tqdm(
        range(0, len(test_examples), batch_size), desc="Self-consistency voting"
    ):
        batch = test_examples[i : i + batch_size]
        questions = [item["question"] for item in batch]

        all_votes = []
        for _ in range(num_samples):
            responses = generate_batch(model, tokenizer, questions)
            for q, response in zip(questions, responses):
                answer = extract_model_answer(response)
                if answer:
                    all_votes.append({"question": q, "answer": answer})

        for q in questions:
            votes = [v["answer"] for v in all_votes if v["question"] == q]
            if votes:
                from collections import Counter

                vote_counts = Counter(votes)
                predicted = vote_counts.most_common(1)[0][0]

                for item in batch:
                    if item["question"] == q:
                        gt = extract_ground_truth(item["answer"], "hashmarks")
                        if answers_match(predicted, gt):
                            correct_count += 1
                        break

    accuracy = correct_count / len(test_examples)

    results = {
        "accuracy": accuracy,
        "num_samples": num_samples,
        "num_eval": num_eval,
        "method": "self_consistency",
    }

    save_json(results, output_file)

    cleanup(model, tokenizer)
    print(f"Q13 complete. Accuracy: {accuracy:.4f}\n")


# ── Main entry point ──


def main():
    parser = argparse.ArgumentParser(description="Part 1: LoRA Fine-Tuning Pipeline")
    parser.add_argument(
        "--question",
        "-q",
        type=int,
        choices=[1, 2, 4, 5, 7, 8, 9, 10, 13],
        help="Run specific question (default: run all)",
    )
    parser.add_argument(
        "--all", action="store_true", default=True, help="Run full pipeline (default)"
    )
    parser.add_argument(
        "--train-7k", action="store_true", help="Include 7k training in Q7"
    )
    parser.add_argument("--adapter-path", type=str, help="Use pre-trained adapter")
    parser.add_argument(
        "--num-eval", type=int, default=100, help="Number of eval samples"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if output exists"
    )

    args = parser.parse_args()

    if args.question:
        args.all = False

    if args.all:
        print("\n" + "=" * 50)
        print("RUNNING FULL PIPELINE (Q1→Q2→Q4→Q5→Q7→Q8→Q9→Q10→Q13)")
        print("=" * 50 + "\n")

        run_q1_baseline_eval(
            num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
        )
        run_q2_extract_failures(force=args.force)
        run_q4_param_count(force=args.force)
        run_q5_train_1k(
            num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
        )
        run_q7_train_3k(
            num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
        )
        run_q7_plot_scaling(force=args.force)
        run_q8_compare_models(force=args.force)
        run_q9_sft_failures(force=args.force)
        run_q10_fewshot(
            num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
        )
        run_q13_open_challenge(
            num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
        )

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50 + "\n")

    else:
        if args.question == 1:
            run_q1_baseline_eval(
                num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
            )
        elif args.question == 2:
            run_q2_extract_failures(force=args.force)
        elif args.question == 4:
            run_q4_param_count(force=args.force)
        elif args.question == 5:
            run_q5_train_1k(
                num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
            )
        elif args.question == 7:
            run_q7_train_3k(
                num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
            )
            run_q7_plot_scaling(force=args.force)
        elif args.question == 8:
            run_q8_compare_models(force=args.force)
        elif args.question == 9:
            run_q9_sft_failures(force=args.force)
        elif args.question == 10:
            run_q10_fewshot(
                num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
            )
        elif args.question == 13:
            run_q13_open_challenge(
                num_eval=args.num_eval, batch_size=args.batch_size, force=args.force
            )


if __name__ == "__main__":
    main()

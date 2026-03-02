"""
Part 1: LoRA SFT for GSM8K Math Problem Solving

This module implements Supervised Fine-Tuning with LoRA for the Qwen model
on the GSM8K dataset for mathematical reasoning.
"""

import re

import gc
import logging
import os
import json
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

from trl import SFTConfig, SFTTrainer
import matplotlib.pyplot as plt

import time

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
TEST_SAMPLES = 100
BATCH_SIZE = 16

SYSTEM_PROMPT = (
    "You are a mathematical problem solver. "
    "Read the problem carefully, work through it step by step, "
    "and provide your final answer inside \\boxed{...}. "
    "Be precise with calculations and show your reasoning clearly."
)

# ── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Device Detection ──────────────────────────────────────────────────────────


def get_device() -> str:
    """
    Auto-detect the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Output Directory Management ───────────────────────────────────────────────


def ensure_output_dir(dirname: str = "outputs") -> str:
    """
    Ensure the output directory exists, create if necessary.

    Args:
        dirname: Name of the output directory (default: "outputs")

    Returns:
        str: Path to the output directory
    """
    os.makedirs(dirname, exist_ok=True)
    return dirname


# ── Model Loading ─────────────────────────────────────────────────────────────


def load_base_model():
    """
    Load the base Qwen2.5-1.5B-Instruct model and tokenizer.

    Returns:
        tuple: (model, tokenizer) loaded and moved to the appropriate device
    """
    device = get_device()
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Target device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer


def load_tokenizer():
    """
    Load just the tokenizer without loading the model.

    Returns:
        AutoTokenizer: The Qwen tokenizer with padding configured
    """
    logger.info(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizer loaded successfully")
    return tokenizer


def load_lora_model(lora_path: str):
    """
    Load the base model with a LoRA adapter.

    Args:
        lora_path: Path to the LoRA adapter checkpoint

    Returns:
        tuple: (model, tokenizer) with LoRA adapter loaded
    """
    device = get_device()
    logger.info(f"Loading base model: {MODEL_NAME}")
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    logger.info(f"Target device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    logger.info(f"Model with LoRA adapter loaded successfully on {device}")
    return model, tokenizer


def cleanup(*objects):
    """
    Delete model/tokenizer objects and free GPU memory.

    Args:
        *objects: Variable number of objects to delete (model, tokenizer, etc.)
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            f"GPU memory freed. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
    else:
        logger.info("Memory cleanup completed")


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_gsm8k_test(n_samples: int = TEST_SAMPLES, seed: int = SEED) -> list[dict]:
    """
    Load GSM8K test dataset samples.

    Args:
        n_samples: Number of samples to select (default: TEST_SAMPLES)
        seed: Random seed for reproducibility (default: SEED)

    Returns:
        list: List of dictionaries with "question" and "answer" keys
    """
    logger.info(f"Loading GSM8K test dataset (n_samples={n_samples}, seed={seed})")

    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n_samples))

    samples = [
        {"question": item["question"], "answer": item["answer"]} for item in dataset
    ]

    logger.info(f"Loaded {len(samples)} test samples")
    return samples


# ── Answer Extraction ─────────────────────────────────────────────────────────


def extract_ground_truth(raw_answer: str) -> str | None:
    """
    Extract the ground-truth answer from GSM8K format.

    GSM8K uses #### to mark the final answer, e.g., "The answer is #### 42"

    Args:
        raw_answer: The raw answer string from the dataset

    Returns:
        str | None: The extracted answer (with commas removed), or None if not found
    """
    match = re.search(r"####\s*(.+)", raw_answer)
    if match:
        # Remove commas for numeric comparison
        return match.group(1).strip().replace(",", "")
    logger.warning(f"No #### marker found in answer: {raw_answer[:50]}...")
    return None


def extract_boxed(text: str) -> str | None:
    """
    Extract content from the last \\boxed{...}, handling nested braces.

    Args:
        text: Text potentially containing \\boxed{...}

    Returns:
        str | None: Content inside the last \\boxed{}, or None if not found
    """
    # Find all \boxed{...} occurrences
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    # Take the last match
    start = matches[-1].end()
    brace_count = 1
    i = start

    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start : i - 1].strip()

    return None


def extract_model_answer(text: str) -> str | None:
    """
    Extract the final answer from model output.

    Tries multiple extraction strategies in order:
    1. \\boxed{...} format (LaTeX)
    2. "The answer is X" pattern
    3. Last number in the response

    Args:
        text: The model's generated response text

    Returns:
        str | None: The extracted answer, or None if extraction failed
    """
    # Strategy 1: Look for \boxed{...}
    boxed = extract_boxed(text)
    if boxed:
        logger.debug(f"Extracted from \\boxed: {boxed}")
        return boxed.replace(",", "")

    # Strategy 2: Look for "The answer is X" or similar patterns
    answer_patterns = [
        r"(?:the\s+)?answer\s+is\s*[:=]?\s*(-?[\d,]+\.?\d*)",
        r"(?:final\s+)?answer[:\s]+(-?[\d,]+\.?\d*)",
        r'(?:therefore|so|thus)[,"\']\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?(-?[\d,]+\.?\d*)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).replace(",", "")
            logger.debug(f"Extracted from pattern '{pattern}': {result}")
            return result

    # Strategy 3: Find the last number in the text
    # Match integers, decimals, and negative numbers
    number_pattern = r"-?[\d,]+\.?\d*"
    numbers = re.findall(number_pattern, text)

    # Filter out numbers that are likely not answers (too short, years, etc.)
    valid_numbers = [
        n.replace(",", "")
        for n in numbers
        if len(n.replace(",", "").replace(".", "").replace("-", "")) > 0
    ]

    if valid_numbers:
        result = valid_numbers[-1]
        logger.debug(f"Extracted last number: {result}")
        return result

    logger.warning(f"Could not extract answer from text: {text[:100]}...")
    return None


# ── LoRA Configuration ─────────────────────────────────────────────────────────


def get_lora_config() -> LoraConfig:
    """
    Create LoRA configuration for Qwen model fine-tuning.

    Returns:
        LoraConfig: Configuration with r=8, alpha=16, dropout=0.05,
                    targeting q_proj, k_proj, v_proj, o_proj
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


# ── Parameter Counting ─────────────────────────────────────────────────────────


def count_parameters(model) -> dict:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model (can be base model or PEFT model with LoRA)

    Returns:
        dict: {
            total_params: int,
            trainable_params: int,
            trainable_pct: float,
            lora_config: dict
        }
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0.0

    # Get LoRA config if model is a PEFT model
    lora_config_dict = {}
    if hasattr(model, "peft_config"):
        # Model has LoRA adapters applied
        active_config = model.active_peft_config
        lora_config_dict = {
            "r": active_config.r,
            "lora_alpha": active_config.lora_alpha,
            "lora_dropout": active_config.lora_dropout,
            "target_modules": list(active_config.target_modules)
            if active_config.target_modules
            else [],
            "bias": active_config.bias,
        }
    else:
        # Use default LoRA config for reference
        config = get_lora_config()
        lora_config_dict = {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
            "bias": config.bias,
        }

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_pct, 4),
        "lora_config": lora_config_dict,
    }


def save_parameter_counts(
    model, output_path: str = "outputs/q4_parameter_counts.json"
) -> None:
    """
    Save parameter counts to JSON file.

    Args:
        model: PyTorch model to count parameters for
        output_path: Path to save the JSON file
    """
    import json

    ensure_output_dir()
    counts = count_parameters(model)

    with open(output_path, "w") as f:
        json.dump(counts, f, indent=2)

    logger.info(f"Parameter counts saved to {output_path}")
    logger.info(
        f"Total: {counts['total_params']:,}, Trainable: {counts['trainable_params']:,}, "
        f"Trainable %: {counts['trainable_pct']:.4f}%"
    )


# ── Batch Generation ───────────────────────────────────────────────────────────


def generate_batch(
    model, tokenizer, questions: list[str], batch_size: int = BATCH_SIZE
) -> list[str]:
    """
    Generate responses for a list of questions in batches.

    Uses chat template formatting with SYSTEM_PROMPT and processes
    questions in batches for efficient inference.

    Args:
        model: The language model (already on device, in eval mode)
        tokenizer: The tokenizer for encoding/decoding
        questions: List of question strings to generate responses for
        batch_size: Number of questions to process per batch (default: BATCH_SIZE)

    Returns:
        list[str]: Generated responses (one per question, in same order)
    """
    all_responses = []
    total_questions = len(questions)

    model.eval()

    for i in range(0, total_questions, batch_size):
        batch_questions = questions[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_questions + batch_size - 1) // batch_size
        logger.info(f"Processing batch {batch_num}/{total_batches}")

        # Build chat-formatted prompts
        prompts = []
        for q in batch_questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # Tokenize batch with padding
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        # Extract only new tokens (skip prompt)
        for j in range(len(batch_questions)):
            new_tokens = outputs[j][prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_responses.append(response)

    logger.info(f"Generated {len(all_responses)} responses")
    return all_responses


def evaluate_model(model, tokenizer, test_data: list[dict]) -> dict:
    """
    Evaluate model on test data, returning accuracy and detailed records.

    Args:
        model: The trained model to evaluate
        tokenizer: The tokenizer for encoding/decoding
        test_data: List of test samples, each with "question" and "answer" keys

    Returns:
        dict: Dictionary containing:
            - accuracy: float (correct/total)
            - correct: int (number of correct predictions)
            - total: int (total number of samples)
            - records: list of dicts with details for each prediction
    """
    logger.info(f"Evaluating {len(test_data)} samples...")

    # Extract questions
    questions = [item["question"] for item in test_data]

    # Generate responses using generate_batch
    responses = generate_batch(model, tokenizer, questions)

    # Evaluate each response
    records = []
    correct = 0
    total = len(test_data)

    for item, response in zip(test_data, responses):
        # Extract model answer and ground truth
        model_answer = extract_model_answer(response)
        ground_truth = extract_ground_truth(item["answer"])

        # Compare (as strings, stripped)
        is_correct = False
        if model_answer is not None and ground_truth is not None:
            is_correct = model_answer.strip() == ground_truth.strip()

        if is_correct:
            correct += 1

        # Record details
        records.append(
            {
                "question": item["question"],
                "response": response,
                "extracted": model_answer,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
        )

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "records": records,
    }


def train_lora_sft(
    model, tokenizer, train_data: list[dict], output_dir: str, epochs: int = 1
) -> float:
    """
    Train model with LoRA using SFTTrainer.

    Args:
        model: Base model to train
        tokenizer: Tokenizer for encoding/decoding
        train_data: List of dicts with "question" and "answer" keys
        output_dir: Directory to save trained adapter
        epochs: Number of training epochs (default: 1)

    Returns:
        float: Training time in seconds
    """
    logger.info(f"Starting LoRA SFT training with {len(train_data)} samples")
    logger.info(f"Output directory: {output_dir}")

    # Format data: convert list of dicts to HuggingFace Dataset
    formatted_data = []
    for item in train_data:
        # Format as chat message with system prompt, user question, and assistant answer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_data.append({"text": text})

    train_dataset = Dataset.from_list(formatted_data)
    logger.info(f"Formatted dataset with {len(train_dataset)} samples")

    # Apply LoRA if not already a PEFT model
    if not isinstance(model, PeftModel):
        logger.info("Applying LoRA configuration to model")
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        max_length=1024,
        completion_only_loss=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train and time
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    logger.info(f"Training completed in {elapsed_time:.2f} seconds")

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Adapter saved to {output_dir}")

    return elapsed_time


def build_few_shot_prompt(question: str, examples: list[dict], n_shots: int = 3) -> str:
    """
    Build few-shot prompt with examples and target question.

    Args:
        question: Target question to answer
        examples: List of dicts with "question" and "answer" keys
        n_shots: Number of few-shot examples to include (default: 3)

    Returns:
        str: Formatted prompt string with few-shot examples
    """
    logger.debug(
        f"Building few-shot prompt with {min(n_shots, len(examples))} examples"
    )

    # Use exactly n_shots examples (or fewer if not enough examples)
    n_examples = min(n_shots, len(examples))

    # Build few-shot examples
    prompt_parts = []
    for i in range(n_examples):
        example = examples[i]
        prompt_parts.append(f"Q: {example['question']}")
        prompt_parts.append(f"A: {example['answer']}")
        prompt_parts.append("")  # Empty line between examples

    # Append target question
    prompt_parts.append(f"Q: {question}")
    prompt_parts.append("A: ")

    return "\n".join(prompt_parts)


def run_baseline_evaluation() -> dict:
    """Evaluate base model on 100 test samples and save baseline results.

    Returns:
        dict: Baseline evaluation results with accuracy and statistics.
    """
    logger.info("Starting baseline evaluation...")

    # Ensure output directory exists
    ensure_output_dir()

    # Load base model
    logger.info("Loading base model...")
    model, tokenizer = load_base_model()

    # Load test data (default 100 samples)
    logger.info("Loading test data...")
    test_dataset = load_gsm8k_test()

    # Evaluate model
    logger.info(f"Evaluating on {len(test_dataset)} samples...")
    results = evaluate_model(model, tokenizer, test_dataset)

    # Save baseline accuracy
    accuracy_path = "outputs/q1_baseline_accuracy.json"
    accuracy_data = {
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
    }
    with open(accuracy_path, "w") as f:
        json.dump(accuracy_data, f, indent=2)
    logger.info(f"Saved baseline accuracy to {accuracy_path}")

    # Extract failure cases
    failures = [r for r in results["records"] if not r["correct"]]
    logger.info(f"Found {len(failures)} failure cases out of {results['total']}")

    # Helper function to extract numeric answer
    def extract_number(answer: str) -> float | None:
        """Extract first numeric value from answer string."""
        if answer is None:
            return None
        # Remove commas and spaces
        cleaned = str(answer).replace(",", "").replace(" ", "")
        # Find first number (including decimals and negative)
        match = re.search(r"-?\d+\.?\d*", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    # Classify each failure
    failure_cases = []
    for record in failures:
        question = record["question"]
        model_response = record["response"]
        extracted_answer = record["extracted"]
        ground_truth = record["ground_truth"]

        # Determine failure type
        failure_type = "misunderstanding"  # default

        # Check for format error
        extracted_num = extract_number(extracted_answer)
        if extracted_num is None:
            failure_type = "format_error"
        else:
            # Check if it's an arithmetic error
            gt_num = extract_number(ground_truth)
            if gt_num is not None:
                # Both are numeric - check if they're close but not equal
                if abs(extracted_num - gt_num) > 0.01:
                    # Check if the numbers suggest an arithmetic operation error
                    # (e.g., similar magnitude but wrong operation)
                    if (
                        abs(extracted_num / (gt_num + 1e-6)) < 10
                        or abs(gt_num / (extracted_num + 1e-6)) < 10
                    ):
                        failure_type = "arithmetic_error"
                    else:
                        failure_type = "reasoning_error"
                else:
                    # Numbers match but marked as incorrect - format issue
                    failure_type = "format_error"
            else:
                # Ground truth not numeric, but extracted is
                failure_type = "reasoning_error"

        failure_cases.append(
            {
                "question": question,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "ground_truth": ground_truth,
                "failure_type": failure_type,
            }
        )

    # Save failure cases
    failures_path = "outputs/q2_failure_cases.json"
    with open(failures_path, "w") as f:
        json.dump(failure_cases, f, indent=2)
    logger.info(f"Saved failure cases to {failures_path}")

    # Cleanup model
    logger.info("Cleaning up model...")
    cleanup(model, tokenizer)

    logger.info("Baseline evaluation complete!")

    return accuracy_data


def save_hyperparameter_info(
    model, output_path: str = "outputs/q4_parameter_counts.json"
) -> None:
    """
    Save both parameter counts and training hyperparameters to JSON file.

    Args:
        model: PyTorch model to count parameters for
        output_path: Path to save the JSON file
    """
    ensure_output_dir()

    # Get parameter counts
    counts = count_parameters(model)

    # Training hyperparameters
    hyperparameters = {
        "learning_rate": 2e-4,
        "batch_size": 8,
        "grad_accum": 4,
        "epochs": 1,
        "max_length": 1024,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }

    # Combine both
    output_data = {
        **counts,
        "hyperparameters": hyperparameters,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Hyperparameter info saved to {output_path}")
    logger.info(
        f"Total: {counts['total_params']:,}, Trainable: {counts['trainable_params']:,}, "
        f"Trainable %: {counts['trainable_pct']:.4f}%"
    )
    logger.info(
        f"Learning rate: {hyperparameters['learning_rate']}, Batch size: {hyperparameters['batch_size']}"
    )


# ── Q5-Q7: SFT Training + Scaling Experiments ───────────────────────────────────


def run_sft_experiments(baseline_accuracy: float = 0.0) -> dict:
    """
    Run SFT training experiments with 1k and 3k samples.

    Trains LoRA adapters on increasing subsets of GSM8K training data,
    evaluates each on the test set, and saves results.

    Args:
        baseline_accuracy: Accuracy of base model (0 training samples).
            Will be filled when run (placeholder 0.0 by default).

    Returns:
        dict: {
            train_sizes: [0, 1000, 3000],
            accuracies: [baseline, acc_1k, acc_3k],
            training_times: [0.0, time_1k, time_3k]
        }
    """
    ensure_output_dir()

    # Load test data once
    test_data = load_gsm8k_test()

    # Load training dataset
    logger.info("Loading GSM8K training dataset...")
    train_dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=SEED)

    # Initialize results structure
    results = {
        "train_sizes": [0, 1000, 3000],
        "accuracies": [baseline_accuracy, 0.0, 0.0],
        "training_times": [0.0, 0.0, 0.0],
    }

    # Experiment configurations
    experiments = [
        {
            "size": 1000,
            "adapter_dir": "outputs/adapters/lora_1k",
            "results_file": "outputs/q5_1k_sft_results.json",
        },
        {
            "size": 3000,
            "adapter_dir": "outputs/adapters/lora_3k",
            "results_file": "outputs/q7_scaling_results.json",
        },
    ]

    for i, exp in enumerate(experiments, start=1):
        size = exp["size"]
        adapter_dir = exp["adapter_dir"]
        results_file = exp["results_file"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Experiment {i}: Training with {size} samples")
        logger.info(f"{'=' * 60}")

        # Ensure adapter directory exists
        os.makedirs(adapter_dir, exist_ok=True)

        # Select training subset
        train_data = train_dataset.select(range(size))
        train_samples = [
            {"question": item["question"], "answer": item["answer"]}
            for item in train_data
        ]

        # Load fresh base model for training
        model, tokenizer = load_base_model()

        # Train LoRA adapter
        training_time = train_lora_sft(
            model=model,
            tokenizer=tokenizer,
            train_data=train_samples,
            output_dir=adapter_dir,
            epochs=1,
        )

        # Cleanup training model
        cleanup(model, tokenizer)

        # Load model with trained adapter for evaluation
        model, tokenizer = load_lora_model(adapter_dir)

        # Evaluate on test set
        eval_results = evaluate_model(model, tokenizer, test_data)
        accuracy = eval_results["accuracy"]

        # Cleanup evaluation model
        cleanup(model, tokenizer)

        # Store results
        results["accuracies"][i] = accuracy
        results["training_times"][i] = training_time

        # Save individual experiment results
        exp_results = {
            "accuracy": accuracy,
            "training_time": training_time,
            "num_examples": size,
            "correct": eval_results["correct"],
            "total": eval_results["total"],
        }
        with open(results_file, "w") as f:
            json.dump(exp_results, f, indent=2)
        logger.info(f"Saved results to {results_file}")

        logger.info(f"Completed {size} sample experiment:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Training time: {training_time:.2f}s")

    # Save final scaling results
    scaling_results_file = "outputs/q7_scaling_results.json"
    with open(scaling_results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved scaling results to {scaling_results_file}")

    return results


def generate_scaling_plot(
    train_sizes: list[int],
    accuracies: list[float],
    training_times: list[float],
    output_path: str = "outputs/q7_scaling_plot.png",
) -> None:
    """
    Generate matplotlib scaling plot with dual y-axes.

    Creates a figure with two subplots:
    1. Accuracy vs Training Size
    2. Training Time vs Training Size

    Args:
        train_sizes: List of training data sizes
        accuracies: List of accuracy values corresponding to each size
        training_times: List of training times in seconds
        output_path: Path to save the plot (default: outputs/q7_scaling_plot.png)
    """
    ensure_output_dir()

    # Create figure with two subplots side by side
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_sizes, accuracies, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Training Data Size", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Model Accuracy vs Training Data Size", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(train_sizes)
    ax1.set_ylim(0, max(accuracies) * 1.1 if max(accuracies) > 0 else 1)

    # Add value labels on points
    for x, y in zip(train_sizes, accuracies):
        ax1.annotate(
            f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Plot 2: Training Time vs Training Size
    ax2.plot(train_sizes, training_times, "r-s", linewidth=2, markersize=8)
    ax2.set_xlabel("Training Data Size", fontsize=12)
    ax2.set_xticks(train_sizes)

    # Add value labels on points
    for x, y in zip(train_sizes, training_times):
        ax2.annotate(
            f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Scaling plot saved to {output_path}")


# ── Q8-Q9: SFT Comparison + Failure Analysis ────────────────────────────────────


def compare_sft_failures() -> list[dict]:
    """
    Compare base model vs SFT-3k on the first 3 failures from Q2.

    Loads baseline failures, re-evaluates both models on these cases,
    and saves comparison results showing whether SFT improved performance.

    Returns:
        list[dict]: Comparison results for each of the 3 cases with:
            - question, base_response, sft_response, ground_truth
            - base_correct, sft_correct
    """
    logger.info("Starting SFT comparison on baseline failures...")
    ensure_output_dir()

    # Load baseline failures from Q2
    failures_path = "outputs/q2_failure_cases.json"
    if not os.path.exists(failures_path):
        logger.error(f"Baseline failures file not found: {failures_path}")
        logger.error("Run baseline evaluation first (Task 10)")
        return []

    with open(failures_path) as f:
        baseline_failures = json.load(f)

    # Take first 3 failures for comparison
    comparison_cases = baseline_failures[:3]
    if len(comparison_cases) < 3:
        logger.warning(
            f"Only {len(comparison_cases)} failures available for comparison"
        )

    logger.info(
        f"Comparing base vs SFT-3k on {len(comparison_cases)} baseline failures"
    )

    # Extract questions for batch processing
    questions = [case["question"] for case in comparison_cases]
    ground_truths = [case["ground_truth"] for case in comparison_cases]

    # Get base model responses
    logger.info("Loading base model for comparison...")
    base_model, base_tokenizer = load_base_model()
    base_responses = generate_batch(base_model, base_tokenizer, questions)
    cleanup(base_model, base_tokenizer)

    # Get SFT-3k model responses
    sft_adapter_path = "outputs/adapters/lora_3k"
    if not os.path.exists(sft_adapter_path):
        logger.error(f"SFT-3k adapter not found: {sft_adapter_path}")
        logger.error("Run SFT training first (Task 12)")
        return []

    logger.info("Loading SFT-3k model for comparison...")
    sft_model, sft_tokenizer = load_lora_model(sft_adapter_path)
    sft_responses = generate_batch(sft_model, sft_tokenizer, questions)
    cleanup(sft_model, sft_tokenizer)

    # Compare results
    comparison_results = []
    for i, case in enumerate(comparison_cases):
        base_extracted = extract_model_answer(base_responses[i])
        sft_extracted = extract_model_answer(sft_responses[i])
        ground_truth = ground_truths[i]

        base_correct = (
            base_extracted is not None
            and ground_truth is not None
            and base_extracted.strip() == ground_truth.strip()
        )
        sft_correct = (
            sft_extracted is not None
            and ground_truth is not None
            and sft_extracted.strip() == ground_truth.strip()
        )

        comparison_results.append(
            {
                "question": case["question"],
                "base_response": base_responses[i],
                "sft_response": sft_responses[i],
                "ground_truth": ground_truth,
                "base_correct": base_correct,
                "sft_correct": sft_correct,
            }
        )

        logger.info(
            f"Case {i + 1}: base_correct={base_correct}, sft_correct={sft_correct}"
        )

    # Save comparison results
    output_path = "outputs/q8_sft_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved SFT comparison to {output_path}")

    return comparison_results


def find_sft_failures() -> list[dict]:
    """
    Find 2 new failure cases specific to SFT-3k model.

    Evaluates SFT-3k on all test samples and identifies failures where
    SFT-3k fails but the base model succeeded (regressions or new failures).

    Returns:
        list[dict]: Up to 2 new SFT-specific failure cases with:
            - question, model_response, extracted_answer, ground_truth, failure_type
    """
    logger.info("Finding new SFT-3k failure cases...")
    ensure_output_dir()

    # Load test data
    test_data = load_gsm8k_test()

    # We need full evaluation records - load test data and get baseline results
    logger.info("Loading base model for comparison...")
    base_model, base_tokenizer = load_base_model()
    base_results = evaluate_model(base_model, base_tokenizer, test_data)
    base_records = {r["question"]: r for r in base_results["records"]}
    cleanup(base_model, base_tokenizer)

    # Load SFT-3k model
    sft_adapter_path = "outputs/adapters/lora_3k"
    if not os.path.exists(sft_adapter_path):
        logger.error(f"SFT-3k adapter not found: {sft_adapter_path}")
        return []

    logger.info("Loading SFT-3k model...")
    sft_model, sft_tokenizer = load_lora_model(sft_adapter_path)
    sft_results = evaluate_model(sft_model, sft_tokenizer, test_data)
    cleanup(sft_model, sft_tokenizer)

    # Helper function to extract numeric answer (same as baseline)
    def extract_number(answer: str) -> float | None:
        if answer is None:
            return None
        cleaned = str(answer).replace(",", "").replace(" ", "")
        match = re.search(r"-?\d+\.?\d*", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    # Find SFT failures where base succeeded
    sft_failures = []
    for sft_record in sft_results["records"]:
        question = sft_record["question"]
        sft_correct = sft_record["correct"]

        # Check if this was a base success
        base_record = base_records.get(question)
        if base_record is None:
            continue

        base_correct = base_record["correct"]

        # We want SFT failures where base succeeded (regressions or new failures)
        if not sft_correct and base_correct:
            # Classify failure type
            extracted_answer = sft_record["extracted"]
            ground_truth = sft_record["ground_truth"]

            failure_type = "misunderstanding"
            extracted_num = extract_number(extracted_answer)

            if extracted_num is None:
                failure_type = "format_error"
            else:
                gt_num = extract_number(ground_truth)
                if gt_num is not None:
                    if abs(extracted_num - gt_num) > 0.01:
                        if (
                            abs(extracted_num / (gt_num + 1e-6)) < 10
                            or abs(gt_num / (extracted_num + 1e-6)) < 10
                        ):
                            failure_type = "arithmetic_error"
                        else:
                            failure_type = "reasoning_error"
                    else:
                        failure_type = "format_error"
                else:
                    failure_type = "reasoning_error"

            sft_failures.append(
                {
                    "question": question,
                    "model_response": sft_record["response"],
                    "extracted_answer": extracted_answer,
                    "ground_truth": ground_truth,
                    "failure_type": failure_type,
                }
            )

    # Take up to 2 failures
    new_failures = sft_failures[:2]
    logger.info(
        f"Found {len(sft_failures)} SFT-specific failures, saving {len(new_failures)}"
    )

    # Save failures
    output_path = "outputs/q9_sft_failures.json"
    with open(output_path, "w") as f:
        json.dump(new_failures, f, indent=2)
    logger.info(f"Saved SFT failures to {output_path}")

    return new_failures


def run_q8_q9_experiments() -> dict:
    """
    Run Q8-Q9 experiments: SFT comparison and failure analysis.

    Returns:
        dict: Results from both experiments
    """
    logger.info("=" * 60)
    logger.info("Running Q8-Q9: SFT Comparison and Failure Analysis")
    logger.info("=" * 60)

    # Q8: Compare base vs SFT-3k on baseline failures
    comparison_results = compare_sft_failures()

    # Q9: Find new SFT-specific failures
    sft_failures = find_sft_failures()

    return {
        "q8_comparison": comparison_results,
        "q9_failures": sft_failures,
    }


def majority_vote(answers: list[str]) -> str:
    """
    Return the most common answer from a list of answers.

    Args:
        answers: List of answer strings

    Returns:
        str: The most common answer, or first answer if tie or empty
    """
    if not answers:
        logger.warning("Empty answer list provided to majority_vote")
        return ""

    if len(answers) == 1:
        return answers[0]

    from collections import Counter

    counter = Counter(answers)
    # Get the most common, return the first one if there's a tie
    most_common = counter.most_common()
    max_count = most_common[0][1]
    # Filter all answers with the maximum count (handle ties)
    tied_answers = [ans for ans, count in most_common if count == max_count]
    return tied_answers[0]  # Return first in case of tie


# ── Q10: Few-Shot Experiments ───────────────────────────────────────────────────


def run_fewshot_experiments() -> dict:
    """
    Run few-shot experiments comparing base and SFT-3k models.

    Evaluates:
    - Base model with 0-shot (baseline)
    - Base model with 3-shot
    - SFT-3k model with 0-shot
    - SFT-3k model with 3-shot

    Returns:
        dict: Results with accuracies and improvements
    """
    logger.info("=" * 60)
    logger.info("Q10: Running Few-Shot Experiments")
    logger.info("=" * 60)
    ensure_output_dir()

    # Load test data
    test_data = load_gsm8k_test()
    questions = [item["question"] for item in test_data]

    # Load few-shot examples from training data
    logger.info("Loading few-shot examples from training data...")
    train_dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=SEED)
    few_shot_examples = [
        {"question": item["question"], "answer": item["answer"]}
        for item in train_dataset.select(range(3))
    ]

    results = {}

    # 1. Base model 0-shot (reuse baseline if available)
    baseline_path = "outputs/q1_baseline_accuracy.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        base_zeroshot = baseline_data["accuracy"]
        logger.info(f"Loaded baseline 0-shot accuracy: {base_zeroshot:.4f}")
    else:
        logger.info("Evaluating base model 0-shot...")
        base_model, base_tokenizer = load_base_model()
        base_results = evaluate_model(base_model, base_tokenizer, test_data)
        base_zeroshot = base_results["accuracy"]
        cleanup(base_model, base_tokenizer)

    results["base_zeroshot"] = base_zeroshot

    # 2. Base model 3-shot
    logger.info("Evaluating base model 3-shot...")
    base_model, base_tokenizer = load_base_model()
    base_3shot_responses = generate_fewshot_batch(
        base_model, base_tokenizer, questions, few_shot_examples, n_shots=3
    )
    base_3shot_correct = 0
    for item, response in zip(test_data, base_3shot_responses):
        model_answer = extract_model_answer(response)
        ground_truth = extract_ground_truth(item["answer"])
        if (
            model_answer
            and ground_truth
            and model_answer.strip() == ground_truth.strip()
        ):
            base_3shot_correct += 1
    base_3shot = base_3shot_correct / len(test_data)
    results["base_3shot"] = base_3shot
    cleanup(base_model, base_tokenizer)
    logger.info(f"Base 3-shot accuracy: {base_3shot:.4f}")

    # 3. SFT-3k 0-shot
    logger.info("Evaluating SFT-3k 0-shot...")
    sft_adapter_path = "outputs/adapters/lora_3k"
    if not os.path.exists(sft_adapter_path):
        logger.error(f"SFT-3k adapter not found: {sft_adapter_path}")
        return results

    sft_model, sft_tokenizer = load_lora_model(sft_adapter_path)
    sft_results = evaluate_model(sft_model, sft_tokenizer, test_data)
    sft_3k_zeroshot = sft_results["accuracy"]
    results["sft_3k_zeroshot"] = sft_3k_zeroshot
    logger.info(f"SFT-3k 0-shot accuracy: {sft_3k_zeroshot:.4f}")

    # 4. SFT-3k 3-shot
    logger.info("Evaluating SFT-3k 3-shot...")
    sft_3shot_responses = generate_fewshot_batch(
        sft_model, sft_tokenizer, questions, few_shot_examples, n_shots=3
    )
    sft_3shot_correct = 0
    for item, response in zip(test_data, sft_3shot_responses):
        model_answer = extract_model_answer(response)
        ground_truth = extract_ground_truth(item["answer"])
        if (
            model_answer
            and ground_truth
            and model_answer.strip() == ground_truth.strip()
        ):
            sft_3shot_correct += 1
    sft_3k_3shot = sft_3shot_correct / len(test_data)
    results["sft_3k_3shot"] = sft_3k_3shot
    cleanup(sft_model, sft_tokenizer)
    logger.info(f"SFT-3k 3-shot accuracy: {sft_3k_3shot:.4f}")

    # Calculate improvements
    base_improvement = (
        ((base_3shot - base_zeroshot) / base_zeroshot * 100) if base_zeroshot > 0 else 0
    )
    sft_improvement = (
        ((sft_3k_3shot - sft_3k_zeroshot) / sft_3k_zeroshot * 100)
        if sft_3k_zeroshot > 0
        else 0
    )

    results["base_improvement"] = f"+{base_improvement:.2f}%"
    results["sft_improvement"] = f"+{sft_improvement:.2f}%"

    # Save results
    output_path = "outputs/q10_fewshot_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved few-shot results to {output_path}")

    return results


def generate_fewshot_batch(
    model, tokenizer, questions: list[str], examples: list[dict], n_shots: int = 3
) -> list[str]:
    """
    Generate responses using few-shot prompting.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to answer
        examples: Few-shot examples with "question" and "answer" keys
        n_shots: Number of few-shot examples to include

    Returns:
        list[str]: Generated responses
    """
    all_responses = []
    total_questions = len(questions)

    model.eval()

    for i in range(0, total_questions, BATCH_SIZE):
        batch_questions = questions[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_questions + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Processing few-shot batch {batch_num}/{total_batches}")

        prompts = []
        for q in batch_questions:
            fewshot_prompt = build_few_shot_prompt(q, examples, n_shots)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fewshot_prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        for j in range(len(batch_questions)):
            new_tokens = outputs[j][prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_responses.append(response)

    logger.info(f"Generated {len(all_responses)} few-shot responses")
    return all_responses


# ── Q12: Aggregate Reflection Data ───────────────────────────────────────────────


def aggregate_reflection_data() -> dict:
    """
    Aggregate failure types from all experiments.

    Loads failure cases from Q2 (baseline) and Q9 (SFT-specific),
    counts failure types, and saves aggregated data.

    Returns:
        dict: Aggregated failure counts
    """
    logger.info("=" * 60)
    logger.info("Q12: Aggregating Reflection Data")
    logger.info("=" * 60)
    ensure_output_dir()

    counts = {
        "arithmetic_errors": 0,
        "reasoning_errors": 0,
        "format_errors": 0,
        "total_failures": 0,
    }

    # Load Q2 baseline failures
    q2_path = "outputs/q2_failure_cases.json"
    if os.path.exists(q2_path):
        with open(q2_path) as f:
            q2_failures = json.load(f)
        logger.info(f"Loaded {len(q2_failures)} failures from Q2")

        for failure in q2_failures:
            failure_type = failure.get("failure_type", "unknown")
            if failure_type == "arithmetic_error":
                counts["arithmetic_errors"] += 1
            elif failure_type == "reasoning_error":
                counts["reasoning_errors"] += 1
            elif failure_type == "format_error":
                counts["format_errors"] += 1
            counts["total_failures"] += 1
    else:
        logger.warning(f"Q2 failure cases not found: {q2_path}")

    # Load Q9 SFT-specific failures
    q9_path = "outputs/q9_sft_failures.json"
    if os.path.exists(q9_path):
        with open(q9_path) as f:
            q9_failures = json.load(f)
        logger.info(f"Loaded {len(q9_failures)} failures from Q9")

        for failure in q9_failures:
            failure_type = failure.get("failure_type", "unknown")
            if failure_type == "arithmetic_error":
                counts["arithmetic_errors"] += 1
            elif failure_type == "reasoning_error":
                counts["reasoning_errors"] += 1
            elif failure_type == "format_error":
                counts["format_errors"] += 1
            counts["total_failures"] += 1
    else:
        logger.warning(f"Q9 failure cases not found: {q9_path}")

    # Save results
    output_path = "outputs/q12_limitation_data.json"
    with open(output_path, "w") as f:
        json.dump(counts, f, indent=2)
    logger.info(f"Saved limitation data to {output_path}")
    logger.info(f"Total failures: {counts['total_failures']}")
    logger.info(f"  Arithmetic: {counts['arithmetic_errors']}")
    logger.info(f"  Reasoning: {counts['reasoning_errors']}")
    logger.info(f"  Format: {counts['format_errors']}")

    return counts


# ── Q13: Open Challenge ──────────────────────────────────────────────────────────

# Store original prompt for comparison
ORIGINAL_SYSTEM_PROMPT = SYSTEM_PROMPT

# Improved prompt with step numbering and clearer format
IMPROVED_SYSTEM_PROMPT = (
    "You are a mathematical problem solver. Follow these steps:\n"
    "1. Read the problem carefully and identify what is being asked.\n"
    "2. Break down the problem into smaller steps.\n"
    "3. Show your work with clear step-by-step calculations.\n"
    "4. Double-check your arithmetic for each step.\n"
    "5. Provide your final answer inside \\boxed{...}.\n\n"
    "Format your response as:\n"
    "Step 1: [first step]\n"
    "Step 2: [second step]\n"
    "...\n"
    "Final Answer: \\boxed{[your answer]}"
)


def run_open_challenge() -> dict:
    """
    Run open challenge with self-consistency and improved prompts.

    Tests 4 ablations:
    1. baseline_sft3k: SFT-3k with original prompt, no self-consistency
    2. new_prompt_only: SFT-3k with improved prompt, no self-consistency
    3. self_consistency_only: SFT-3k with original prompt, self-consistency
    4. combined: SFT-3k with improved prompt + self-consistency

    Returns:
        dict: Results with ablation accuracies and extraction success rates
    """
    logger.info("=" * 60)
    logger.info("Q13: Running Open Challenge")
    logger.info("=" * 60)
    ensure_output_dir()

    # Load test data
    test_data = load_gsm8k_test()
    questions = [item["question"] for item in test_data]

    # Load SFT-3k model
    sft_adapter_path = "outputs/adapters/lora_3k"
    if not os.path.exists(sft_adapter_path):
        logger.error(f"SFT-3k adapter not found: {sft_adapter_path}")
        return {}

    sft_model, sft_tokenizer = load_lora_model(sft_adapter_path)

    results = {
        "hypothesis": "Self-consistency and structured prompting improve accuracy by reducing random errors and format issues.",
        "method": "Self-consistency with n=5 samples at temperature=0.7, majority voting, and improved prompt with step numbering.",
        "ablations": {},
    }

    # Track extraction success rates
    original_extraction_success = 0
    new_extraction_success = 0

    # 1. Baseline SFT-3k (original prompt, no self-consistency)
    logger.info("Ablation 1: Baseline SFT-3k...")
    baseline_responses = generate_batch_with_prompt(
        sft_model, sft_tokenizer, questions, ORIGINAL_SYSTEM_PROMPT
    )
    baseline_correct, baseline_extracted = evaluate_responses(
        test_data, baseline_responses
    )
    original_extraction_success += baseline_extracted
    results["ablations"]["baseline_sft3k"] = baseline_correct / len(test_data)
    logger.info(f"Baseline SFT-3k: {results['ablations']['baseline_sft3k']:.4f}")

    # 2. New prompt only
    logger.info("Ablation 2: New prompt only...")
    new_prompt_responses = generate_batch_with_prompt(
        sft_model, sft_tokenizer, questions, IMPROVED_SYSTEM_PROMPT
    )
    new_prompt_correct, new_prompt_extracted = evaluate_responses(
        test_data, new_prompt_responses
    )
    new_extraction_success += new_prompt_extracted
    results["ablations"]["new_prompt_only"] = new_prompt_correct / len(test_data)
    logger.info(f"New prompt only: {results['ablations']['new_prompt_only']:.4f}")

    # 3. Self-consistency only (original prompt)
    logger.info("Ablation 3: Self-consistency only...")
    sc_responses = generate_self_consistency_batch(
        sft_model, sft_tokenizer, questions, ORIGINAL_SYSTEM_PROMPT, n_samples=5
    )
    sc_correct, sc_extracted = evaluate_responses(test_data, sc_responses)
    original_extraction_success += sc_extracted
    results["ablations"]["self_consistency_only"] = sc_correct / len(test_data)
    logger.info(
        f"Self-consistency only: {results['ablations']['self_consistency_only']:.4f}"
    )

    # 4. Combined (new prompt + self-consistency)
    logger.info("Ablation 4: Combined...")
    combined_responses = generate_self_consistency_batch(
        sft_model, sft_tokenizer, questions, IMPROVED_SYSTEM_PROMPT, n_samples=5
    )
    combined_correct, combined_extracted = evaluate_responses(
        test_data, combined_responses
    )
    new_extraction_success += combined_extracted
    results["ablations"]["combined"] = combined_correct / len(test_data)
    logger.info(f"Combined: {results['ablations']['combined']:.4f}")

    cleanup(sft_model, sft_tokenizer)

    # Calculate best accuracy and improvement
    best_accuracy = max(results["ablations"].values())
    baseline_acc = results["ablations"]["baseline_sft3k"]
    improvement = (
        ((best_accuracy - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    )

    results["best_accuracy"] = best_accuracy
    results["improvement_over_sft3k"] = f"+{improvement:.2f}%"

    # Calculate extraction success rates (per response, so divide by 2 for each prompt type)
    total_questions = len(test_data)
    results["extraction_success_original_prompt"] = original_extraction_success / (
        2 * total_questions
    )
    results["extraction_success_new_prompt"] = new_extraction_success / (
        2 * total_questions
    )

    # Save results
    output_path = "outputs/q13_open_challenge.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved open challenge results to {output_path}")

    return results


def generate_batch_with_prompt(
    model, tokenizer, questions: list[str], system_prompt: str
) -> list[str]:
    """
    Generate responses with a custom system prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions
        system_prompt: Custom system prompt to use

    Returns:
        list[str]: Generated responses
    """
    all_responses = []
    total_questions = len(questions)

    model.eval()

    for i in range(0, total_questions, BATCH_SIZE):
        batch_questions = questions[i : i + BATCH_SIZE]

        prompts = []
        for q in batch_questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        for j in range(len(batch_questions)):
            new_tokens = outputs[j][prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_responses.append(response)

    return all_responses


def generate_self_consistency_batch(
    model, tokenizer, questions: list[str], system_prompt: str, n_samples: int = 5
) -> list[str]:
    """
    Generate responses using self-consistency (majority voting).

    Samples n_samples responses per question with temperature=0.7,
    extracts answers, and uses majority voting.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions
        system_prompt: System prompt to use
        n_samples: Number of samples for self-consistency

    Returns:
        list[str]: Final responses with majority-voted answers
    """
    all_responses = []
    total_questions = len(questions)

    model.eval()

    for i in range(0, total_questions, BATCH_SIZE):
        batch_questions = questions[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_questions + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Self-consistency batch {batch_num}/{total_batches}")

        # For each question, sample n_samples times
        for q in batch_questions:
            sample_answers = []
            sample_responses = []

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            for _ in range(n_samples):
                inputs = tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)
                prompt_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )

                new_tokens = outputs[0][prompt_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                sample_responses.append(response)

                answer = extract_model_answer(response)
                if answer:
                    sample_answers.append(answer)

            # Use majority vote for final answer
            if sample_answers:
                voted_answer = majority_vote(sample_answers)
                # Return the response that contains the voted answer
                for resp in sample_responses:
                    extracted = extract_model_answer(resp)
                    if extracted and extracted.strip() == voted_answer.strip():
                        all_responses.append(resp)
                        break
                else:
                    all_responses.append(sample_responses[0])
            else:
                all_responses.append(sample_responses[0] if sample_responses else "")

    return all_responses


def evaluate_responses(test_data: list[dict], responses: list[str]) -> tuple[int, int]:
    """
    Evaluate responses against ground truth.

    Args:
        test_data: List of test samples with "answer" key
        responses: List of model responses

    Returns:
        tuple: (correct_count, extraction_success_count)
    """
    correct = 0
    extracted = 0

    for item, response in zip(test_data, responses):
        model_answer = extract_model_answer(response)
        ground_truth = extract_ground_truth(item["answer"])

        if model_answer is not None:
            extracted += 1

        if (
            model_answer
            and ground_truth
            and model_answer.strip() == ground_truth.strip()
        ):
            correct += 1

    return correct, extracted


def main():
    """
    Main orchestrator - runs ALL Part 1 experiments automatically.

    Total runtime: ~5-7 hours on T4 GPU
    """
    logger.info("=" * 70)
    logger.info("PART 1: LoRA SFT for GSM8K - Full Experiment Suite")
    logger.info("=" * 70)
    logger.info("This will run ALL experiments automatically.")
    logger.info("Estimated total runtime: 5-7 hours on T4 GPU")
    logger.info("=" * 70)

    start_time = time.time()

    # PHASE 1: BASELINE EVALUATION (Q1-Q2)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Baseline Evaluation (Q1-Q2)")
    logger.info("=" * 70)
    logger.info("Estimated time: 20-25 minutes")

    baseline_results = run_baseline_evaluation()
    baseline_accuracy = baseline_results["accuracy"]
    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")

    # PHASE 2: PARAMETER COUNTS (Q4)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Parameter Counts (Q4)")
    logger.info("=" * 70)
    logger.info("Estimated time: 2-3 minutes")

    model, tokenizer = load_base_model()
    save_hyperparameter_info(model)
    cleanup(model, tokenizer)

    # PHASE 3: SFT TRAINING EXPERIMENTS (Q5-Q7)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: SFT Training Experiments (Q5-Q7)")
    logger.info("=" * 70)
    logger.info("Estimated time: 3-4 hours total")

    sft_results = run_sft_experiments(baseline_accuracy=baseline_accuracy)

    logger.info("\nGenerating scaling plot...")
    generate_scaling_plot(
        sft_results["train_sizes"],
        sft_results["accuracies"],
        sft_results["training_times"],
    )

    # PHASE 4: FAILURE ANALYSIS (Q8-Q9)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Failure Analysis (Q8-Q9)")
    logger.info("=" * 70)
    logger.info("Estimated time: 30-40 minutes")

    run_q8_q9_experiments()

    # PHASE 5: FEW-SHOT EXPERIMENTS (Q10)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Few-Shot Experiments (Q10)")
    logger.info("=" * 70)
    logger.info("Estimated time: 40-50 minutes")

    run_fewshot_experiments()

    # PHASE 6: REFLECTION DATA AGGREGATION (Q12)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Reflection Data Aggregation (Q12)")
    logger.info("=" * 70)
    logger.info("Estimated time: <1 minute")

    aggregate_reflection_data()

    # PHASE 7: OPEN CHALLENGE (Q13)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: Open Challenge (Q13)")
    logger.info("=" * 70)
    logger.info("Estimated time: 1-2 hours")

    run_open_challenge()

    # COMPLETION SUMMARY
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"Total runtime: {hours}h {minutes}m")
    logger.info("\nGenerated files:")
    logger.info("  outputs/q1_baseline_accuracy.json")
    logger.info("  outputs/q2_failure_cases.json")
    logger.info("  outputs/q4_parameter_counts.json")
    logger.info("  outputs/q5_1k_sft_results.json")
    logger.info("  outputs/q7_scaling_results.json")
    logger.info("  outputs/q7_scaling_plot.png")
    logger.info("  outputs/q8_sft_comparison.json")
    logger.info("  outputs/q9_sft_failures.json")
    logger.info("  outputs/q10_fewshot_results.json")
    logger.info("  outputs/q12_limitation_data.json")
    logger.info("  outputs/q13_open_challenge.json")
    logger.info("  outputs/adapters/lora_1k/")
    logger.info("  outputs/adapters/lora_3k/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

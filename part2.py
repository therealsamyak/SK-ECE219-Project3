import json
import random
import re
import argparse
import signal
import sys
import io
import traceback
from pathlib import Path

import builtins

import pandas as pd
import numpy as np
import torch
from pydantic import BaseModel, Field


class TimeoutError(Exception):
    """Custom timeout exception for Executor."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout.

    Args:
        signum: Signal number (always signal.SIGALRM)
        frame: Current stack frame (unused)
    """
    raise TimeoutError("Code execution timed out")


class Executor:
    """Executes Python code with isolated namespace and timeout protection.

    Provides sandboxed execution environment with:
    - Timeout protection via signal.SIGALRM (900s default = 15 minutes)
    - Isolated namespace (only pd, np, df, print available)
    - Captured stdout/stderr
    - State reset between tasks
    """

    def __init__(self, df: pd.DataFrame, timeout: int = 900):
        """Initialize Executor with DataFrame and timeout.

        Args:
            df: DataFrame to make available in namespace as 'df'
            timeout: Execution timeout in seconds (default: 900 = 15 minutes)
        """
        self.df = df
        self.timeout = timeout
        self._namespace = None
        self.reset()

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check if code contains dangerous patterns.

        Args:
            code: Python code to check

        Returns:
            tuple: (is_safe, error_message)
        """
        dangerous_patterns = [
            (r"\bimport\s+os\b", "Importing os module is not allowed"),
            (r"\bimport\s+subprocess\b", "Importing subprocess module is not allowed"),
            (r"\bimport\s+sys\b", "Importing sys module is not allowed"),
            (r"\bfrom\s+os\s+import", "Importing from os module is not allowed"),
            (
                r"\bfrom\s+subprocess\s+import",
                "Importing from subprocess module is not allowed",
            ),
            (r"\bfrom\s+sys\s+import", "Importing from sys module is not allowed"),
            (
                r"\b(?:rm|rd|rmdir|del|delete)\s+",
                "File deletion commands are not allowed",
            ),
            (r"\bopen\s*\(", "File operations (open) are not allowed"),
            (r"\b__import__\s*\(", "Dynamic imports are not allowed"),
            (r"\beval\s*\(", "eval() is not allowed for security reasons"),
            (r"\bexec\s*\(", "exec() is not allowed for security reasons"),
            (r"\bcompile\s*\(", "compile() is not allowed for security reasons"),
            (r"\brequests\.", "Internet access via requests is not allowed"),
            (r"\burllib\.", "Internet access via urllib is not allowed"),
            (r"\bhttp\.", "Internet access via http module is not allowed"),
            (r"\bsubprocess\.", "Subprocess calls are not allowed"),
            (r"\bos\.", "Direct os module access is not allowed"),
            (r"\bsys\.", "Direct sys module access is not allowed"),
            (r"\bglobals\s*\(\)", "Access to globals() is not allowed"),
            (r"\blocals\s*\(\)", "Access to locals() is not allowed"),
            (r"\bvars\s*\(\)", "Access to vars() is not allowed"),
            (r"\b__builtins__", "Access to __builtins__ is not allowed"),
            (r"\bgetattr\s*\(", "Use of getattr() is restricted"),
            (r"\bsetattr\s*\(", "Use of setattr() is restricted"),
            (r"\bdelattr\s*\(", "Use of delattr() is restricted"),
        ]

        code_lower = code.lower()
        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, code_lower):
                return False, error_msg

        return True, ""

    def reset(self):
        """Clear namespace state for a new task."""
        # Create fresh namespace with only safe modules
        self._namespace = {
            "__builtins__": builtins.__dict__,
            "pd": pd,
            "np": np,
            "df": self.df,
            "print": print,
        }

    def run(self, code: str) -> tuple[str, str]:
        """Execute code and return captured stdout/stderr.

        Args:
            code: Python code to execute

        Returns:
            tuple: (stdout, stderr) strings

        Note:
            - TimeoutError caught and returned in stderr
            - SyntaxError caught and returned in stderr
            - RuntimeError caught and returned in stderr
            - Traceback printed to stderr buffer, not raised
            - Dangerous code patterns are blocked before execution
        """
        is_safe, error_msg = self._check_code_safety(code)
        if not is_safe:
            stdout = ""
            stderr = f"SecurityError: {error_msg}. Code execution blocked for safety."
            return stdout, stderr

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf

        old_signal = signal.signal(signal.SIGALRM, _timeout_handler)
        timeout_triggered = False

        try:
            signal.alarm(int(self.timeout))
            exec(code, self._namespace, self._namespace)
        except TimeoutError:
            timeout_triggered = True
            print(
                f"TimeoutError: Code execution exceeded {self.timeout} seconds",
                file=stderr_buf,
            )
            traceback.print_exc(file=stderr_buf)
        except SyntaxError:
            traceback.print_exc(file=stderr_buf)
        except Exception:
            traceback.print_exc(file=stderr_buf)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_signal)
            sys.stdout, sys.stderr = old_stdout, old_stderr

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        if timeout_triggered:
            stderr += f"\n[Executor] Timeout of {self.timeout}s enforced"

        return stdout, stderr


# Constants
QUESTIONS_PATH = "datasets/share_data/da-dev-questions.jsonl"
LABELS_PATH = "datasets/share_data/da-dev-labels.jsonl"
TABLES_DIR = "datasets/share_data/da-dev-tables"
OUTPUT_DIR = "outputs"

# Selected task IDs
SELECTED_IDS = [0, 5, 9, 10, 14, 18, 24, 25, 26, 55]


# Pydantic Schemas for Structured Output
class PlannerOutput(BaseModel):
    """Planner agent output with structured response."""

    thought: str = Field(min_length=10, max_length=500)
    is_done: bool
    response: str = Field(min_length=1)


class ObservationOutput(BaseModel):
    """Observer agent output for execution results."""

    summary: str = Field(max_length=500)
    extracted_values: dict[str, str]
    error_type: str | None


def load_jsonl(filepath: str) -> list[dict]:
    """Load JSONL file into list of dictionaries.

    Args:
        filepath: Path to JSONL file.

    Returns:
        List of dictionaries, one per line.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_json(data, filename: str):
    """Save data to outputs/{filename}, pretty-printed.

    Args:
        data: Data to save (dict, list, etc.).
        filename: Name of the file to save (relative to OUTPUT_DIR).
    """
    filepath = Path(OUTPUT_DIR) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


# Global model instance (singleton pattern)
_model = None
_tokenizer = None
_outlines_model = None


def load_model():
    """Load Qwen3-4B-Instruct model with 4-bit quantization.

    Returns:
        tuple: (model, tokenizer, outlines_model)

    Uses singleton pattern to avoid reloading.
    """
    global _model, _tokenizer, _outlines_model

    if _model is not None:
        return _model, _tokenizer, _outlines_model

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import outlines

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Qwen3-4B-Instruct-2507 model with 4-bit quantization...")
    _model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", quantization_config=bnb_config, device_map="auto"
    )
    _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    _outlines_model = outlines.from_transformers(_model, _tokenizer)
    print("Model loaded successfully.")

    return _model, _tokenizer, _outlines_model


def run_q14():
    """Q14: Count records and print one example from each file.

    Saves:
        - q14_counts.json: {"num_questions": int, "num_labels": int}
    """
    print("\n=== Q14: Dataset Inspection ===")

    questions = load_jsonl(QUESTIONS_PATH)
    labels = load_jsonl(LABELS_PATH)

    counts = {"num_questions": len(questions), "num_labels": len(labels)}

    print(f"Questions: {counts['num_questions']}")
    print(f"Labels: {counts['num_labels']}")

    print("\nExample question:")
    print(f"  Keys: {list(questions[0].keys())}")
    print(f"  {json.dumps(questions[0], indent=2)}")

    print("\nExample label:")
    print(f"  Keys: {list(labels[0].keys())}")
    print(f"  {json.dumps(labels[0], indent=2)}")

    save_json(counts, "q14_counts.json")
    print("Q14 complete.\n")


def run_q15():
    """Q15: Pick 3 random question IDs, load their CSVs.

    Saves:
        - q15_examples.json: List of 3 entries with shape/dtypes/head/question
    """
    print("\n=== Q15: CSV File Inspection ===")

    questions = load_jsonl(QUESTIONS_PATH)
    _labels = load_jsonl(LABELS_PATH)  # noqa: F841 — loaded for completeness

    q_map = {q["id"]: q for q in questions}

    random.seed(42)
    sampled_ids = random.sample(list(q_map.keys()), 3)

    examples = []
    for q_id in sampled_ids:
        q_record = q_map[q_id]
        csv_filename = q_record["file_name"]
        csv_path = Path(TABLES_DIR) / csv_filename

        df = pd.read_csv(csv_path)

        example = {
            "id": q_id,
            "file_name": csv_filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head(3).to_dict(orient="records"),
            "question": q_record["question"],
        }
        examples.append(example)

        print(f"\nID {q_id}: {csv_filename}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:5]}...")
        print(f"  Question: {q_record['question'][:80]}...")

    save_json(examples, "q15_examples.json")
    print("\nQ15 complete.\n")


def run_q16():
    """Q16: Find 2 examples with multiple @name[value] slots in format field.

    Saves:
        - q16_format_examples.json: 2 multi-part answer examples
    """
    print("\n=== Q16: Multi-Part Answer Format ===")

    questions = load_jsonl(QUESTIONS_PATH)

    multi_slot_questions = []
    pattern = r"@\w+\["

    for q in questions:
        matches = re.findall(pattern, q["format"])
        if len(matches) >= 2:
            multi_slot_questions.append(q)

    print(f"Found {len(multi_slot_questions)} questions with multiple slots")

    selected = multi_slot_questions[:2]

    examples = []
    for q in selected:
        example = {
            "id": q["id"],
            "question": q["question"],
            "format": q["format"],
            "file_name": q["file_name"],
            "level": q["level"],
            "num_slots": len(re.findall(pattern, q["format"])),
        }
        examples.append(example)

        print(f"\nID {q['id']}:")
        print(f"  Question: {q['question'][:80]}...")
        print(f"  Format: {q['format']}")
        print(f"  Slots: {example['num_slots']}")

    explanation = {
        "representation": "Multi-part answers use @name[value] slots where 'name' identifies the answer part and 'value' is the expected value. For example, @mean_fare_child[34.56] and @mean_fare_adult[42.78] represent two separate values for different age groups.",
        "examples": examples,
    }

    save_json(explanation, "q16_format_examples.json")
    print("\nQ16 complete.\n")


def run_q17():
    """Q17: Print the 10 SELECTED_IDS tasks.

    Saves:
        - q17_selected_tasks.json: List of 10 selected task details
    """
    print("\n=== Q17: Selected Tasks ===")

    questions = load_jsonl(QUESTIONS_PATH)

    q_map = {q["id"]: q for q in questions}

    selected_tasks = []
    for task_id in SELECTED_IDS:
        q_record = q_map[task_id]
        task_info = {
            "id": q_record["id"],
            "question": q_record["question"],
            "concepts": q_record["concepts"],
            "file_name": q_record["file_name"],
            "level": q_record["level"],
        }
        selected_tasks.append(task_info)

        print(f"\n{task_id}: {q_record['question'][:60]}...")
        print(f"  Level: {q_record['level']}")
        print(f"  File: {q_record['file_name']}")

    save_json(selected_tasks, "q17_selected_tasks.json")
    print("\nQ17 complete.\n")


class ReActAgent:
    """ReAct agent with Coder and Observer components.

    Coder: Generates Python code using unstructured generation
    Observer: Summarizes execution output using structured generation
    """

    def __init__(self):
        """Initialize agent and load model."""
        self.model, self.tokenizer, self.outlines_model = load_model()

    def _format_prompt(self, text: str) -> str:
        """Wrap text in chat template for Qwen3-Instruct."""
        messages = [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def coder(self, instruction: str, question: str, context: str) -> str:
        """Generate Python code for instruction using unstructured generation.

        Args:
            instruction: What code to generate
            question: The original question being answered
            context: Context about the DataFrame and available data

        Returns:
            str: Python code (extracted from markdown if wrapped)
        """
        print(f"[Coder] Generating code for instruction: {instruction[:80]}...")
        print(f"[Coder] Question: {question[:60]}...")

        prompt = f"""You are a Python data analysis code generator. Your job is to write EXECUTABLE Python code, not answer strings.

Question: {question}

Context: {context}

Instruction: {instruction}

CRITICAL RULES:
1. Write ONLY valid, executable Python code using pandas (pd) and numpy (np)
2. The DataFrame is available as variable 'df' - use it directly
3. pd, np, and df are already imported - DO NOT write import statements
4. DO NOT use answer format strings like @name[value] in your code
5. Use print() to output computed values
6. For statistical tests, use scipy.stats if needed (already available)

EXAMPLES OF CORRECT CODE:
```python
mean_fare = df['Fare'].mean()
print(f"Mean fare: {{mean_fare:.2f}}")
```

```python
from scipy import stats
stat, p_value = stats.normaltest(df['bmi'])
print(f"P-value: {{p_value:.4f}}")
```

EXAMPLES OF INCORRECT CODE (DO NOT WRITE THIS):
- @mean_fare[34.65]  ← This is NOT Python code, it's an answer format
- mean = 34.65  ← Don't hardcode values, compute them from data

Now write the Python code to perform the instruction:"""

        print(f"[Coder] Prompt length: {len(prompt)} chars")

        prompt = self._format_prompt(prompt)

        device = (
            self.model.device
            if hasattr(self.model, "device") and self.model.device is not None
            else "cpu"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        print("[Coder] Generating from model (max_new_tokens=500, temperature=0.7)...")

        outputs = self.model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=500,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        print("[Coder] Model generation complete")

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated_text = re.sub(r"<\|[^>]+\|>", "", raw).strip()

        code = self._extract_code_from_markdown(generated_text)

        print(f"[Coder] Final code length: {len(code)} chars")

        return code

    def _extract_code_from_markdown(self, text: str) -> str:
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        if re.search(r"@\w+\[", text) and not text.strip().startswith("print"):
            return "print('Error: Generated answer format instead of code. Please generate executable Python code.')"

        return text.strip()

    def observer(self, stdout: str, stderr: str, instruction: str) -> ObservationOutput:
        """Summarize execution output into structured observation.

        Args:
            stdout: Standard output from execution (truncated to 500 chars)
            stderr: Standard error from execution (truncated to 500 chars)
            instruction: The instruction that was executed

        Returns:
            ObservationOutput: Structured observation with summary, values, error type
        """
        print("[Observer] Summarizing execution output")
        print(f"[Observer]   Instruction: {instruction[:60]}...")

        stdout_truncated = stdout[:500] if len(stdout) > 500 else stdout
        stderr_truncated = stderr[:500] if len(stderr) > 500 else stderr

        print(
            f"[Observer]   Stdout length: {len(stdout)} chars (truncated to {len(stdout_truncated)})"
        )
        print(
            f"[Observer]   Stderr length: {len(stderr)} chars (truncated to {len(stderr_truncated)})"
        )

        if stdout_truncated or stderr_truncated:
            context = f"Instruction: {instruction}\n\nStdout:\n{stdout_truncated}\n\nStderr:\n{stderr_truncated}"
        else:
            context = f"Instruction: {instruction}\n\nNo output produced."

        prompt = f"""You are an execution observer. Summarize what happened when code was executed.

{context}

Provide:
- summary: Brief description of what happened (max 500 chars)
- extracted_values: Dictionary of any key values found (e.g., {{"mean_age": "32.5"}})
- error_type: Type of error if any, or null if successful"""

        print("[Observer] Generating structured observation...")
        prompt = self._format_prompt(prompt)
        result_json = self.outlines_model(prompt, ObservationOutput, max_new_tokens=200)
        result = json.loads(result_json)

        print(f"[Observer]   Summary: {result['summary'][:100]}...")
        print(f"[Observer]   Error type: {result.get('error_type')}")
        print(
            f"[Observer]   Extracted values: {list(result.get('extracted_values', {}).keys())}"
        )

        return ObservationOutput(**result)

    def planner(self, question: str, context: str, history: list) -> PlannerOutput:
        """Plan next step based on question, context, and execution history.

        Args:
            question: The original question being answered
            context: Context about the DataFrame and available data
            history: List of previous (thought, instruction, code, observation) tuples

        Returns:
            PlannerOutput: Structured output with thought, is_done, and response
        """
        print("[Planner] Planning next step")
        print(f"[Planner]   Question: {question[:60]}...")

        history_text = ""
        if history:
            print(f"[Planner]   History size: {len(history)} steps")
            history_items = []
            for i, (thought, instruction, code, observation) in enumerate(
                history[-3:], 1
            ):
                history_items.append(
                    f"Step {i}:\n"
                    f"  Thought: {thought}\n"
                    f"  Instruction: {instruction[:200]}\n"
                    f"  Code: {code[:200]}\n"
                    f"  Observation: {observation[:300]}"
                )
            history_text = "\n\n".join(history_items)
            print(f"[Planner]   Using last {min(3, len(history))} steps from history")
        else:
            print("[Planner]   No history yet (first step)")

        prompt = f"""You are a data analysis planner. Given a question and execution history, decide what to do next.

Question: {question}

Context: {context}
"""

        if history_text:
            prompt += f"""
Previous Steps:
{history_text}
"""

        prompt += """
Respond with a structured output containing:
- thought: Your reasoning about the current state and what to do next (10-500 characters)
- is_done: Whether you have enough information to answer the question (true/false)
- response: Your response based on is_done:
  * If is_done=true: Provide the FINAL ANSWER using the EXACT required format from context
    - Use ONLY the @name[value] format shown in "Required Answer Format"
    - Do NOT add explanatory text, descriptions, or extra words
    - Example: @mean_fare[34.65] NOT "The mean fare is @mean_fare[34.65]"
    - Example: @price_range_mean[16.65] @price_range_median[15.67] @price_range_std_dev[6.72]
  * If is_done=false: Provide the next instruction for the coder to execute

CRITICAL: When is_done=true, response must contain ONLY the formatted answer, nothing else."""

        print("[Planner] Generating structured plan...")
        prompt = self._format_prompt(prompt)
        result_json = self.outlines_model(prompt, PlannerOutput, max_new_tokens=300)
        result = json.loads(result_json)

        print(f"[Planner]   Thought: {result['thought'][:100]}...")
        print(f"[Planner]   is_done: {result['is_done']}")
        print(f"[Planner]   Response: {result['response'][:100]}...")

        return PlannerOutput(**result)

    def _build_context(
        self, df: pd.DataFrame, question: str, constraints: str, answer_format: str = ""
    ) -> str:
        context_parts = [
            f"DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            f"Data types: {df.dtypes.to_dict()}",
            f"First 3 rows:\n{df.head(3).to_string()}",
            f"\nQuestion: {question}",
            f"\nConstraints: {constraints}",
        ]
        if answer_format:
            context_parts.append(f"\nRequired Answer Format: {answer_format}")
        return "\n".join(context_parts)

    def run(
        self,
        question: str,
        constraints: str,
        df: pd.DataFrame,
        max_steps: int = 5,
        answer_format: str = "",
    ) -> tuple[str, list]:
        print(f"[ReAct] Starting ReAct loop (max {max_steps} steps)")
        print(f"[ReAct] Question: {question[:100]}...")
        print(f"[ReAct] Constraints: {constraints[:100]}...")
        executor = Executor(df)
        context = self._build_context(df, question, constraints, answer_format)
        history = []

        for step in range(max_steps):
            print(f"[ReAct] Step {step + 1}/{max_steps}: Starting planner")
            planner_out = self.planner(question, context, history)
            print(f"[ReAct]   Planner thought: {planner_out.thought[:100]}...")
            print(f"[ReAct]   Planner is_done: {planner_out.is_done}")

            if planner_out.is_done:
                print(
                    f"[ReAct]   Final answer from planner: {planner_out.response[:200]}..."
                )
                print(f"[ReAct] Step {step + 1}/{max_steps}: Complete (is_done=True)")
                return planner_out.response, history

            instruction = planner_out.response[:2000]
            print(f"[ReAct]   Planner instruction: {instruction[:100]}...")
            original_instruction = instruction

            for retry in range(3):
                print(f"[ReAct]   Code generation attempt {retry + 1}/3")
                code = self.coder(instruction, question, context)
                code = code[:2000]
                print(f"[ReAct]     Generated code length: {len(code)} chars")
                print(f"[ReAct]     Code preview: {code[:150]}...")

                print(f"[ReAct]   Executing code (timeout: {executor.timeout}s)...")
                stdout, stderr = executor.run(code)
                print("[ReAct]     Execution complete")
                print(f"[ReAct]       Stdout length: {len(stdout)} chars")
                print(f"[ReAct]       Stderr length: {len(stderr)} chars")
                if stderr:
                    print(f"[ReAct]       Stderr preview: {stderr[:200]}...")

                print("[ReAct]   Running observer...")
                observation = self.observer(stdout, stderr, instruction)
                observation_summary = observation.summary[:500]
                print(f"[ReAct]     Observer summary: {observation_summary[:100]}...")
                print(f"[ReAct]     Observer error_type: {observation.error_type}")

                history.append(
                    (planner_out.thought, instruction, code, observation_summary)
                )

                if observation.error_type is None:
                    print("[ReAct]   Execution successful, moving to next step")
                    break
                else:
                    print(f"[ReAct]   Error detected: {observation.error_type}")
                    if retry < 2:
                        print("[ReAct]   Retrying with enhanced error feedback...")
                        error_details = f"{observation.error_type}"
                        if "SyntaxError" in stderr:
                            error_details += " - The code has invalid Python syntax. Remember to write executable Python code, not answer format strings like @name[value]."
                        elif "NameError" in stderr:
                            error_details += " - A variable or function is not defined. Make sure to use only pd, np, df, and print."
                        elif "KeyError" in stderr:
                            error_details += " - Column name not found. Check available columns in the DataFrame."
                        instruction = f"{original_instruction}\n\nPrevious error: {error_details}\nStderr: {stderr[:300]}\n\nPlease write valid, executable Python code."
                    else:
                        print("[ReAct]   Max retries reached, moving to next step")

        print(f"[ReAct] Max steps ({max_steps}) reached, generating final response")
        final_planner = self.planner(question, context, history)
        print(f"[ReAct] Final answer from planner: {final_planner.response[:200]}...")
        print("[ReAct] ReAct loop complete")
        return final_planner.response, history

    def evaluate_answer(self, predicted: str, ground_truth: list) -> bool:
        pattern = r"@(\w+)\[([^\]]+)\]"
        predicted_matches = re.findall(pattern, predicted)
        predicted_dict = {name: value.strip() for name, value in predicted_matches}

        ground_truth_dict = {name: value for name, value in ground_truth}

        if set(predicted_dict.keys()) != set(ground_truth_dict.keys()):
            return False

        for name in predicted_dict:
            pred_val = predicted_dict[name]
            truth_val = ground_truth_dict[name]

            try:
                pred_float = float(pred_val)
                truth_float = float(truth_val)
                rel_tol = 0.011
                abs_tol = 0.011
                if not (
                    abs(pred_float - truth_float)
                    <= max(rel_tol * max(abs(pred_float), abs(truth_float)), abs_tol)
                ):
                    return False
            except ValueError:
                if pred_val.lower() != truth_val.lower():
                    return False

        return True


def test_planner_output(prompt: str) -> dict:
    """Test structured output generation with PlannerOutput schema.

    Args:
        prompt: Input prompt for the planner.

    Returns:
        dict: PlannerOutput as dictionary.
    """
    _, tokenizer, outlines_model = load_model()

    full_prompt = f"""You are a data analysis planner. Given a question, decide what to do next.

{prompt}

Respond with a structured output containing:
- thought: Your reasoning (10-500 characters)
- is_done: Whether you have enough information to answer (true/false)
- response: What to do next (the instruction for the coder, or final answer if done)"""

    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    result_json = outlines_model(full_prompt, PlannerOutput, max_new_tokens=200)
    result = json.loads(result_json)
    return result


def run_q18():
    """Q18: Test structured output with 5 prompts (including 1 where is_done=true).

    Saves:
        - q18_demo.json: 5 prompts with PlannerOutput results
    """
    print("\n=== Q18: Structured Output Demo ===")

    prompts = [
        "The user asks: What is the average age of passengers?",
        "The user asks: How many passengers survived?",
        "The user asks: What is the correlation between age and fare?",
        "The user asks: Show me the distribution of passenger classes.",
        "Calculate the mean fare: @mean_fare[32.20] (final answer)",  # This should result in is_done=true
    ]

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt[:60]}...")
        output = test_planner_output(prompt)
        print(f"  is_done: {output['is_done']}")
        print(f"  response: {output['response'][:60]}...")

        results.append(
            {
                "prompt": prompt,
                "output": output,
            }
        )

    demo = {
        "model": "Qwen/Qwen3-4B-Instruct",
        "quantization": "4-bit NF4",
        "num_prompts": len(prompts),
        "prompts": results,
    }

    save_json(demo, "q18_demo.json")
    print("\nQ18 complete.\n")


def run_q20():
    print("\n=== Q20: Full ReAct Agent Evaluation ===")

    questions = load_jsonl(QUESTIONS_PATH)
    labels = load_jsonl(LABELS_PATH)

    q_map = {q["id"]: q for q in questions}
    l_map = {label["id"]: label for label in labels}

    agent = ReActAgent()

    results = []
    traces = []

    correct = 0
    total = 0

    for task_id in SELECTED_IDS:
        print(f"\nProcessing task {task_id}...")

        q_record = q_map[task_id]
        l_record = l_map[task_id]

        csv_path = Path(TABLES_DIR) / q_record["file_name"]
        df = pd.read_csv(csv_path)

        final_answer, history = agent.run(
            question=q_record["question"],
            constraints=q_record["constraints"],
            df=df,
            max_steps=5,
            answer_format=q_record["format"],
        )

        is_correct = agent.evaluate_answer(final_answer, l_record["common_answers"])

        if is_correct:
            correct += 1
        total += 1

        trace = {
            "task_id": task_id,
            "question": q_record["question"],
            "history": [
                {
                    "thought": thought,
                    "instruction": instruction,
                    "code": code,
                    "observation": observation,
                }
                for thought, instruction, code, observation in history
            ],
            "final_answer": final_answer,
            "ground_truth": l_record["common_answers"],
            "is_correct": is_correct,
        }
        traces.append(trace)

        results.append(
            {
                "task_id": task_id,
                "is_correct": is_correct,
                "final_answer": final_answer,
                "ground_truth": l_record["common_answers"],
            }
        )

        print(f"  Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")

    accuracy = correct / total if total > 0 else 0.0

    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

    success_trace = next((t for t in traces if t["is_correct"]), None)
    failure_trace = next((t for t in traces if not t["is_correct"]), None)
    recovery_trace = next(
        (
            t
            for t in traces
            if any("error" in h["observation"].lower() for h in t["history"])
        ),
        None,
    )

    representative_traces = []
    if success_trace:
        representative_traces.append(success_trace)
    if failure_trace:
        representative_traces.append(failure_trace)
    if recovery_trace:
        representative_traces.append(recovery_trace)
    elif len(traces) > len(representative_traces):
        representative_traces.append(traces[0])

    accuracy_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }

    save_json(accuracy_data, "q20_accuracy.json")
    save_json(representative_traces[:3], "q20_traces.json")

    print("\nQ20 complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Part 2: ReAct Agent Implementation")
    parser.add_argument("--q14", action="store_true", help="Run only Q14")
    parser.add_argument("--q15", action="store_true", help="Run only Q15")
    parser.add_argument("--q16", action="store_true", help="Run only Q16")
    parser.add_argument("--q17", action="store_true", help="Run only Q17")
    parser.add_argument("--q18", action="store_true", help="Run only Q18")
    parser.add_argument("--q20", action="store_true", help="Run only Q20")

    args = parser.parse_args()

    if not (args.q14 or args.q15 or args.q16 or args.q17 or args.q18 or args.q20):
        print("\n" + "=" * 70)
        print("RUNNING FULL PIPELINE (Q14→Q15→Q16→Q17→Q18→Q20)")
        print("=" * 70 + "\n")
        run_q14()
        run_q15()
        run_q16()
        run_q17()
        run_q18()
        run_q20()
        print("\n" + "=" * 70)
        print("FULL PIPELINE COMPLETE")
        print("=" * 70 + "\n")
    else:
        if args.q14:
            run_q14()
        if args.q15:
            run_q15()
        if args.q16:
            run_q16()
        if args.q17:
            run_q17()
        if args.q18:
            run_q18()
        if args.q20:
            run_q20()


if __name__ == "__main__":
    main()

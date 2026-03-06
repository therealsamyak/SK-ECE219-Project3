# Part B: Agentic Data Mining with ReAct

## Introduction

Modern data mining pipelines increasingly operate at a scale where manual curation is no longer feasible. In many real-world settings—customer logs, scientific datasets, public records, transaction streams, and web data—the raw material is noisy, heterogeneous, and only partially structured. Traditionally, teams would write domain-specific feature extraction and analytics code by hand, but today we increasingly rely on AI agents to help discover, clean, transform, and analyze data automatically.

This sub-project explores a practical question in modern large-scale data mining:

_Can we build an LLM-based agent that answers data analysis questions by interacting with real CSV tables, while producing outputs that are reliable enough to be evaluated automatically?_

The project is motivated by a practical tension:

- LLMs are flexible, but their outputs can be unpredictable and hard to parse.
- Data analysis requires precision: a single wrong filter, join, or aggregation can flip an answer.
- Large-scale settings demand automation: if we cannot automate the whole workflow, the system does not scale.

## What You Will Build

You will build a small data analysis agentic system that solves questions about CSV files. The system follows the ReAct paradigm (Reasoning + Acting): it iteratively (i) decides what to do next, (ii) writes Python code to perform that step, (iii) executes the code, (iv) summarizes the tool output into a structured observation, and (v) uses that observation to continue or stop with a final answer.

You will also incorporate structured output (via Outlines), enabling robust orchestration and automatic evaluation.

## Learning Objectives

By the end of this project, you should be able to:

- Work with a real benchmark for evaluating LLM agents on data analysis tasks.
- Implement a basic agentic system with planning, tool-use (code), and observations.
- Use structured generation to guarantee parseable intermediate outputs.
- Evaluate an agent quantitatively and analyze typical failure modes (format errors, reasoning errors, code bugs, data issues).

## Dataset: InfiAgent-DABench (DAEval)

We use the InfiAgent-DABench benchmark. Each example consists of:

- A natural-language question about a CSV table;
- Additional constraints (e.g., rounding rules, filtering rules);
- A required output format (so answers can be checked automatically);
- A reference CSV file (one of many, with different schemas);
- Ground-truth answers in a closed-form representation.

The dataset is provided in this link to you in the following directory layout:

- `da-dev-questions.jsonl` (questions + metadata)
- `da-dev-labels.jsonl` (ground-truth answers)
- `da-dev-tables/` (CSV files referenced by questions)

**Important:** The CSV files do not share a common schema. One question may reference a finance table with columns like `revenue` and `quarter`, while another may reference a sports play-by-play table with completely different fields. This is a realistic large-scale data mining challenge: heterogeneous schemas are the norm, not the exception.

## Why Structured Output? (Outlines)

A typical pipeline has multiple components: a planner that decides the next step, a coder that writes code, and an executor that runs code. If the planner emits free-form text like:

> I think we should compute the mean fare and round to two decimals.

then your program must guess what to do next. This is not ideal: the agent might change phrasing across runs, break your parser, or produce ambiguous instructions.

Instead, we will use structured output so the planner emits a machine-readable object, for example:

```json
{ "thought": "...", "is_done": false, "response": "Compute the mean of Fare." }
```

In this project, you will use Outlines to constrain an open-source LLM to generate outputs that follow a Pydantic schema. This eliminates a major source of agent instability: parsing errors.

## Agents and Agentic Data Mining

In this project, an agent is a system that:

1. maintains state (what it has done so far),
2. decides what to do next,
3. interacts with external tools (here: Python execution over CSV files),
4. uses observations from tools to update its next decision,
5. terminates when it can produce a final answer.

Compared to a single one-shot LLM response, an agent can:

- **maintain and manage context:** keep a persistent working memory (tool outputs, intermediate results, constraints), selectively summarize it, and feed back only the most relevant state when the context window is limited (this is especially important when we are using smaller models);
- **use specialized prompting per role:** separate prompts for planning, coding, and verification, so each step is more focused than a single all-in-one prompt;
- **decompose complex tasks** into smaller steps,
- **ground answers in actual computations** (reducing hallucinations),
- **debug when code fails** (recover from errors and retry),
- **provide transparent traces** (what it attempted and what it observed).

## The ReAct Idea (Reasoning + Acting)

ReAct interleaves two things:

- **Reasoning:** decide what to do next given the current state and past observations.
- **Acting:** invoke an external tool (here: execute Python code over CSV files).

In our setting, the action is Python code that reads a CSV and computes statistics. After execution, instead of directly feeding the raw stdout/stderr back to the planner, we introduce an additional Observation agent that converts tool output into a concise, task-relevant summary of observation (e.g., key numbers, detected schema issues, error type, and suggested fix) (We bring in observation agent to avoid context window become too big for certain complex tasks). The planner then uses this structured observation to decide the next step or terminate with the final formatted answer (e.g., `@mean fare[34.65]`).

## Setup and Starter Code

You may work in Google Colab or locally. Your solution must be runnable from a clean environment. We recommend:

- Python ≥ 3.10
- `transformers`, `accelerate`, `torch`
- `pandas`, `numpy`
- `pydantic`
- `outlines`

**Compute note:** A 4B-parameter open model is typically feasible on a single GPU (e.g., Colab T4/L4/A100), but generation can still be slow.

**Safety note:** Executing LLM-generated code can be dangerous. In this project:

- Do not allow internet access from the execution environment.
- Restrict code to reading the provided CSV files only.
- Avoid writing files, deleting files, or calling system commands.

## Model Requirement

Use `Qwen/Qwen3-4B-Instruct-2507` (or any model under 5B parameters). You should load the model once and reuse it for all roles in your pipeline. Different agents (Planner/Coder/Observer) are implemented by different prompts and may use different generation paradigms (e.g., structured outputs via Outlines + Pydantic for Planner and Observer, code-focused prompting for Coder).

## Task 1 — Dataset Inspection and Sanity Checks

Before building an agent, let's understand the dataset format.

### Task 1.1 Load and Inspect JSONL Files

**QUESTION 14:** Load `da-dev-questions.jsonl` and `da-dev-labels.jsonl`. Report:

- The number of questions and labels.
- The set of keys present in a question record (print one example).
- The set of keys present in a label record (print one example).

### Task 1.2 Inspect the Table References

**QUESTION 15:** Pick 3 random question IDs. For each:

- Print the file name of the referenced CSV.
- Load the CSV with pandas and print `df.shape`, `df.dtypes`, and `df.head(3)`.
- Print the corresponding question.

### Task 1.3 Understand the Required Answer Format

**QUESTION 16:** Find 2 examples where the required format contains multiple answer slots (e.g., two or more `@name[value]` fields). Explain:

- How the dataset represents multi-part answers in the labels.
- How you plan to evaluate such answers automatically.

### Task 1.4 Checking the subset

**QUESTION 17:** Unfortunately, the model we are going to use is still not powerful enough to solve all the tasks. Here we are selecting 10 sub-tasks that are proved to be solvable:

The selected IDs are: `SELECTED_IDS = [0, 5, 9, 10, 14, 18, 24, 25, 26, 55]`.

- Print out and check those tasks.

## Task 2 — Model Loading and Structured Output: Make the Planner Parseable

Please refer to the helper code for reference.

First, let's load the model we are going to use: `MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"`.

Next, build a structured Planner component that returns a JSON object following a Pydantic schema such as:

```python
class PlannerOutput(BaseModel):
    thought: str
    is_done: bool
    response: str
```

You must use Outlines (or an equivalent constrained decoding approach) so that the JSON is always valid. You can refer to the helper code for how to use Outlines.

**QUESTION 18:** Demonstrate (with 5 different prompts) that your planner always returns valid JSON that parses into your Pydantic model without try/except fallbacks. Include at least one case where the planner decides it is done (`is_done=true`).

**QUESTION 19:** Explain in a few sentences why structured output is useful for large-scale data mining pipelines.

## Task 3 — Build a ReAct Data Analysis Agent

Implement a 4-part system:

- **Planner (Reasoning):** decides the next analysis step, or stops with the final formatted answer.
- **Coder (Action):** writes Python code for the planner's instruction.
- **Executor (Tool):** runs the code and returns raw stdout/stderr (or an error trace).
- **Observer (Structured Observation):** converts raw tool output into a concise, task-relevant observation that the planner can reliably use.

Your agent must run a loop for up to `max_steps=5` iterations:

1. Planner produces structured output (`thought`, `is_done`, `response`).
2. If `is_done=true`, return `response` as the final answer.
3. Otherwise, Coder writes Python code based on Planner's response and Executor runs it to obtain raw stdout/stderr.
4. Observer reads the raw output and the planner's original response and produces an observation summary (e.g., extracted values, warnings, error category, next-step hint).
5. Append the quadruple (instruction, code, raw output, observation summary) to the history and continue.

### Required Features

- **Error recovery:** if the executor returns an error (e.g., parsing error, missing column, type conversion, runtime exception), the system must use the error message as feedback and attempt to fix the code and retry, instead of immediately failing.
- **Bounded context (runnable on Google Colab):** the system must avoid unbounded growth of the prompt/history. In particular, it should remain runnable on a single Tesla T4 GPU with 15 GB GPU memory. Please optimize your design of the agents, especially the observation summarization agent.
- **Structured outputs for planner:** planner must return a machine-parseable object with a fixed schema (recommended: Pydantic + Outlines).

**QUESTION 20:** Run your ReAct agent on the 10 tasks. Report:

- Accuracy
- At least 3 qualitative traces (planner thought, code, observation, final answer) that illustrate interesting behaviors: success, failure, recovery from an error.

## Part B: Building a ReAct-Style Data Analysis Agent

### Introduction

In Part A, you fine-tuned a small language model to improve its mathematical reasoning on structured benchmarks. But real-world data analysis rarely looks like a clean benchmark: data lives in CSV files, questions are open-ended, and getting to an answer requires writing code, running it, inspecting results, and iterating.

This part of the project asks you to build a **ReAct-style agent** — a system in which a language model interleaves *reasoning* and *acting* in a loop. At each step, the model can either call a tool (e.g., run a Python snippet on a CSV file) or emit a final answer. The agent keeps running until it produces an answer or hits a step budget.

The central challenge here is not raw model capability — you will use GPT-4.1-mini as the backbone — but **reliable agent design under real constraints**: bounded context windows, tool call failures, and open-ended questions that require multi-step data exploration.

---

### What is a ReAct Agent?

ReAct [4] is a prompting and agent design pattern in which the model generates alternating *Thought* and *Action* steps:

```
Thought: I need to find the average sale price grouped by neighborhood.
Action: run_python("df.groupby('Neighborhood')['SalePrice'].mean().sort_values()")
Observation: Neighborhood
            CollgCr    197965.77
            Veenker    238772.73
            ...
Thought: I have the grouped means. The highest is Veenker.
Answer: The neighborhood with the highest average sale price is Veenker.
```

Each tool call produces an *observation* that is appended to the context, and the model uses that observation to decide what to do next. This loop continues until the model emits an `Answer:` tag or the step budget is exhausted.

---

### Dataset: Ames Housing

You will use the **Ames Housing dataset** [5], a widely used tabular dataset describing residential home sales in Ames, Iowa. It contains 1,460 rows and 81 columns, including numerical features (lot area, year built, above-ground living area) and categorical features (neighborhood, building type, sale condition). The prediction target is `SalePrice`.

You will not be building a predictive model here. Instead, your agent will answer open-ended analytical questions about this dataset by dynamically generating and executing Python code.

---

### System Design

Your agent should implement the following loop:

```
while steps < MAX_STEPS:
    response = llm(messages)
    if "Answer:" in response:
        return extract_answer(response)
    elif "Action:" in response:
        tool_output = execute_tool(response)
        messages.append(tool_output as Observation)
    else:
        break
```

#### Core Components

**LLM backbone:** Use GPT-4.1-mini via the OpenAI API (`gpt-4.1-mini`). You will call this model at each step of the agent loop.

**Tool: `run_python`:** The only tool your agent needs is a Python code execution tool. It takes a string of Python code, runs it in a restricted local namespace (with `pandas`, `numpy`, and `scipy` pre-imported and a pre-loaded `df` variable pointing to the Ames Housing dataframe), and returns the output (stdout + result of last expression) as a string. Execution errors should be caught and returned as the observation so the model can self-correct.

**Context management:** At each step, the full message history (system prompt, question, all prior Thought/Action/Observation turns) is sent to the model. For short agent runs this is fine, but for longer chains you should be aware that observation outputs can grow large. You may truncate long tool outputs (e.g., to 2,000 characters) to avoid hitting the context window limit.

**Step budget:** Set `MAX_STEPS = 10` as the default. If the agent does not emit an `Answer:` within 10 steps, it should return a fallback response.

---

### Prompt Design

The system prompt is the most important design decision in your agent. It should:

1. Describe the ReAct format precisely (Thought → Action → Observation → ... → Answer).
2. Specify the exact syntax for tool calls so they can be parsed reliably.
3. Tell the model what tools are available and what the `df` variable contains.
4. Instruct the model to include clear reasoning in every Thought step.

A suggested tool call syntax:

```
Action: run_python
```python
<your code here>
```
```

Your parser should extract the code block following `Action: run_python` and execute it.

---

### Tasks

#### Task 6 — Build and Validate the Agent

**QUESTION 14: (20 points)** Implement the ReAct agent as described above. Your implementation should include:

- The agent loop with step budget,
- The `run_python` tool with error handling,
- A system prompt that clearly specifies the ReAct format,
- Context truncation for long observations.

Run your agent on the following warm-up questions and report the full agent trace (all Thought/Action/Observation steps) for each:

1. How many rows and columns does the dataset have?
2. What is the mean and standard deviation of `SalePrice`?
3. Which neighborhood has the highest median sale price?

**QUESTION 15: (10 points)** Inspect the agent traces from Question 14. For each question:

- How many steps did the agent take?
- Did it arrive at the correct answer?
- Were there any tool errors or unexpected behaviors?

---

#### Task 7 — Multi-Step Data Analysis

Now stress-test your agent with questions that require multiple tool calls and intermediate reasoning.

**QUESTION 16: (30 points)** Run your agent on the following questions and report the full agent trace and final answer for each:

1. What are the top 5 features most correlated with `SalePrice` (by absolute Pearson correlation)?
2. Is there a statistically significant difference in mean `SalePrice` between houses with and without a central air conditioning system (`CentralAir` column)? Report the test statistic and p-value.
3. Among houses built after 1990, what fraction have a garage capacity (`GarageCars`) of 2 or more?

**QUESTION 17: (10 points)** For the correlation analysis in Question 16.1:

- Identify which features are numerical vs. categorical.
- Explain how your agent handled (or should have handled) the presence of categorical features in a Pearson correlation analysis.
- If your agent made an error here, describe what went wrong and how you would fix it.

---

#### Task 8 — Robustness and Failure Analysis

**QUESTION 18: (20 points)** Design two adversarial questions that are likely to challenge your agent — for example, questions involving missing data, ambiguous column names, or multi-step aggregations with filtering conditions. Run your agent on these questions and report:

- The full agent trace,
- Whether the agent recovered from any tool errors,
- The final answer and whether it is correct.

**QUESTION 19: (10 points)** Reflection: Based on your experiments in Tasks 6–8, what are the two most common failure modes of your ReAct agent? For each, describe:

- What the failure looks like in the trace,
- Why it happens (root cause),
- One concrete fix you could implement.
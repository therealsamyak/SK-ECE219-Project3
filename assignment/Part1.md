## Part A: Teaching a Small Model to Reason: LoRA Fine-Tuning, Knowledge

### Introduction

State-of-the-art language models capable of complex reasoning (e.g., GPT-5, Claude, DeepSeek-R1) contain hundreds of billions of parameters, require expensive GPU clusters to serve, and are often accessible only through paid APIs. Yet many practical settings (on-device inference, cost-sensitive deployment at scale, classroom use, and privacy-constrained environments) demand models that are small, fast, and cheap to run. The catch is that small models, while efficient, are typically much worse at multi-step reasoning: they may understand the problem statement but fail at planning, intermediate computation, or faithful execution.

This project asks a concrete question: Can we teach a small model to reason more like a large one, without training from scratch? Training a new model end-to-end is unrealistic in an academic setting because it requires massive compute (thousands of GPU-hours) and enormous pre-training corpora (trillions of tokens). Instead, we start with a strong pre-trained or instruction-tuned model that already understands language and then adapt it to a target behavior using **Supervised Fine-Tuning (SFT)**. In SFT, the model is shown examples of the desired behavior (here: clear, step-by-step math solutions) and is trained to reproduce that style reliably.

A natural concern is cost. Even if we avoid pre-training, fine-tuning a 1.5B-parameter model can still be expensive if we update all weights. Full fine-tuning also risks **catastrophic forgetting**, where the model loses general capabilities while specializing on a narrow task. This motivates parameter-efficient methods that (i) train only a tiny fraction of parameters and (ii) leave the original model weights untouched, while still allowing the model to learn a new skill.

---

### Parameter-Efficient Fine-Tuning: Adapting Without Rewriting the Model

Parameter-efficient fine-tuning (PEFT) methods modify how we represent an "update" to the model. Instead of changing the full weight matrices in each transformer layer, we introduce a small set of additional trainable parameters that steer the frozen model. This has three practical benefits:

- **Compute efficiency:** fewer trainable parameters means faster training and lower GPU memory use.
- **Storage and sharing:** only small adapter files need to be saved and distributed.
- **Stability:** freezing the base model reduces catastrophic forgetting and preserves general language competence.

In this project, we use one of the most widely adopted PEFT techniques: **Low-Rank Adaptation (LoRA)** [2].

---

### Low-Rank Adaptation (LoRA)

LoRA is based on an empirical observation: the weight updates that matter for fine-tuning are often approximately _low-rank_. Rather than learning a full update ΔW ∈ ℝ^(d×d) to a weight matrix W, LoRA represents the update as a product of two thin matrices:

$$h = (W + \Delta W)x = Wx + BAx, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times d},\ r \ll d$$

Here, W is the original (frozen) weight matrix, and only the LoRA parameters A and B are trained. This changes the number of trainable parameters per adapted layer from d² to 2dr, which can be orders of magnitude smaller when r ≪ d. Concretely, for a model like Qwen2.5-1.5B with rank r = 8 applied to the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), this corresponds to training roughly **0.14% of the total parameters** — about a ~710× reduction in trainable parameters compared to full fine-tuning.

Practically, LoRA gives us a clean experimental knob: we can increase adapter capacity by increasing the rank r, or trade off compute versus performance by selecting which modules to adapt. Throughout this project, we use the Hugging Face PEFT library [6], which provides a simple interface for attaching LoRA adapters to transformer models and training only those adapter parameters.

---

### Why Math Reasoning?

Mathematical reasoning is an ideal testbed for studying model behavior because:

- **Answers are unambiguous.** Unlike open-ended generation, a math answer is either correct or incorrect, so evaluation is straightforward.
- **Multi-step reasoning is required.** Even grade-school word problems demand decomposition, intermediate computations, and a final synthesis step.
- **Failures are diagnosable.** When a model gets the wrong answer, we can often localize the error (e.g., arithmetic slip, flawed logic, or misunderstanding the question).

---

### The GSM8K Dataset

We use GSM8K (Grade School Math 8K) [1], a widely-used benchmark of grade-school level math word problems. The dataset contains 7,473 training examples and 1,319 test examples. Each problem requires 2–8 steps of elementary arithmetic and reasoning, and the answer is always an integer. Here is a representative example:

> **GSM8K Example**
>
> **Question:** Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
>
> **Answer:** In the beginning, Betty has only 100/2 = $50. Betty's grandparents gave her 15 × 2 = $30. This means Betty needs 100 − 50 − 30 − 15 = $5 more. **5**

---

### Our Model: Qwen2.5-1.5B-Instruct

Throughout this project, we use Qwen2.5-1.5B-Instruct [3], a small instruction-tuned language model from Alibaba with 1.5 billion parameters. It has been pre-trained on a large text corpus and instruction-tuned for general chat — but it has not been specifically trained for mathematical reasoning. This makes it an ideal "student" model: competent at language but weak at math.

---

### The Gap We Want to Close

To see the problem concretely, consider the following question from the GSM8K test set, and how our base model handles it compared to a strong model (GPT-4.1-mini):

**Base Model (Qwen2.5-1.5B-Instruct) — INCORRECT**

> **Question:** Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
>
> **Model's reasoning (excerpt):**
> Total feed = Morning feed + Afternoon feed
> 15 + 25 = 7 cups ← Elementary arithmetic error!
> ...
> Final meal = 60 − 7 = 53
>
> **Model's answer:** 53 (Ground truth: 20)

The model sets up the problem correctly (each chicken gets 3 cups, 20 chickens → 60 cups total) but then makes a catastrophic arithmetic error: 15 + 25 = 7. This cascades through the rest of the solution.

Now consider GPT-4.1-mini solving a similar problem:

**GPT-4.1-mini — CORRECT**

> **Question:** Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
>
> **GPT's reasoning:**
>
> 1. The wallet costs 100.
> 2. Betty has half: 100/2 = 50
> 3. Parents give her 15.
> 4. Grandparents give 2 × 15 = 30
> 5. Total: 50 + 15 + 30 = 95
> 6. Still needs: 100 − 95 = 5
>
> **Answer: 5 ✓**

Notice the difference: GPT's solution is clearly structured, uses numbered steps, performs correct arithmetic, and presents the answer in a clean format. **The goal of this project is to teach our small model to reason like this — through fine-tuning, prompting strategies, and knowledge distillation.**

---

### Project Roadmap

- **Task 1:** Benchmark the base model to establish the performance floor.
- **Task 2:** Apply LoRA fine-tuning with GSM8K training data and study scaling behavior.
- **Task 3:** Investigate few-shot prompting and its interaction with fine-tuning.
- **Task 4:** Discover that data quality matters more than data quantity through knowledge distillation, and attempt to push accuracy further.

---

### Setup and Evaluation Protocol

All experiments in this project follow a common evaluation protocol to ensure results are comparable across models and training setups.

#### System Prompt Guidelines

Because GSM8K problems require multi-step reasoning, we strongly recommend using a system prompt during evaluation (and matching it during training). In practice, a well-designed system prompt often improves reliability by enforcing a consistent reasoning style and a predictable final-answer format that can be parsed automatically.

Your system prompt should do two things:

1. **Elicit structured reasoning:** promote step-by-step work and intermediate calculations.
2. **Encourage a consistent answer template:** ask for a clearly marked final answer (using a stable marker) so automated extraction is reliable.

In practice, adding lightweight formatting constraints often improves both (i) reasoning clarity and (ii) evaluation robustness. Common, easy-to-parse conventions include:

- Ending with a clearly labeled final answer such as `Answer: <answer>` or `Final: <answer>`.
- Using a single LaTeX marker like `\boxed{...}` for the final answer.

We encourage you to experiment with 2–3 prompt variants early (e.g., different answer markers or step formatting), then **choose one prompt format and keep it fixed** for the remainder of the project. Keeping the prompt fixed ensures results remain directly comparable across models and training regimes.

#### Evaluation Protocol

- **Test set:** 100 questions from the GSM8K test split (fixed across all experiments for compute efficiency).
- **Metric:** Accuracy = (number of correct answers) / (total questions).

#### Compute Environment

Note: It is highly recommended you apply for Colab Pro with your student IDs right away, as this project requires heavy GPU usage!

All experiments can be run on Google Colab. Approximate runtimes on a T4 GPU:

- Evaluation (100 questions): ~20 minutes
- LoRA training (1,000 examples, 1 epoch): ~40–50 minutes
- LoRA training (3,000 examples, 1 epoch): ~80–90 minutes
- LoRA training (7,473 examples, 1 epoch): ~3–4 hours

---

### Task 1 — Baseline: How Good is the Base Model?

Before any fine-tuning, we first measure how well the base Qwen2.5-1.5B-Instruct model solves GSM8K problems out of the box. This establishes a performance floor that all subsequent interventions (fine-tuning, prompting, distillation) will be compared against.

#### Task 1.1 Baseline Evaluation

Evaluate the base model on a fixed subset of 100 GSM8K test questions using the project's standardized prompting and answer-extraction setup.

Note: To reduce inference time, use batching. A batch size of 16 is recommended (and works well on Colab). With batching, 100 questions typically take about 25 minutes. As always, test performance and confirm your batching syntax on a small sample before running the full code.

**QUESTION 1: (10 points)** Run the base Qwen2.5-1.5B-Instruct model on 100 GSM8K test questions and report the accuracy. You should expect approximately 35–40%. (Exact values may vary slightly depending on the prompt format and extraction rule.)

**QUESTION 2: (10 points)** Inspect at least 3 cases where the base model produces an incorrect answer. For each example, include:

- The question,
- A short excerpt of the model's solution highlighting the failure,
- The extracted answer vs. the ground-truth answer.

Classify each failure mode (e.g., arithmetic slip, reasoning/logical error, misunderstanding the problem, formatting/extraction issue). Do you observe any recurring patterns?

---

### Task 2 — LoRA Fine-Tuning on GSM8K

The base model struggles with math. Can we improve it by showing it thousands of worked examples? In this task, you will apply LoRA SFT using the GSM8K training split, starting small and scaling up to study how data quantity affects performance.

#### Task 2.1 Understanding the Training Pipeline

Before training anything, take time to inspect the helper code and understand what is actually being optimized. LoRA SFT is conceptually simple, but many failures in practice come from not understanding the data formatting, where adapters are applied, and which tokens contribute to the loss.

Conceptually, the training pipeline has three key ingredients:

1. **Data formatting:** Each GSM8K training example is converted into a chat-style interaction (system prompt + user question + assistant solution). The assistant solution should follow a consistent answer convention so evaluation can extract answers reliably.
2. **LoRA configuration:** LoRA adapters are inserted into selected linear layers (commonly attention projections and MLP projections). The base weights remain frozen, and only the adapter parameters are trained.
3. **Completion-only loss:** The loss is computed only on the assistant's response tokens (not on the system or user tokens). This focuses training on producing better solutions rather than memorizing the prompt.

Default hyperparameters are summarized below. Use these unless a question explicitly asks you to experiment.

| Hyperparameter        | Default        | Notes                              |
| --------------------- | -------------- | ---------------------------------- |
| LoRA rank (r)         | 8              | Controls adapter capacity          |
| LoRA alpha            | 16             | Scaling factor (rule of thumb: 2r) |
| LoRA dropout          | 0.05           |                                    |
| Learning rate         | 2 × 10⁻⁴       | Cosine schedule with 5% warmup     |
| Epochs                | 1              |                                    |
| Per-device batch size | 8              |                                    |
| Gradient accumulation | 4              | Effective batch size = 32          |
| Max sequence length   | 1024           |                                    |
| Target modules        | Attention only | q,k,v,o proj                       |

**QUESTION 3: (15 points)** Pick the three hyperparameters (LoRA rank, LoRA alpha, Gradient accumulation) from the table above and explain:

- what each hyperparameter controls,
- what you expect to happen if you increase it,
- what you expect to happen if you decrease it.

Your answer should reflect practical tradeoffs (e.g., compute/memory, stability, overfitting vs. underfitting).

**QUESTION 4: (15 points)** Report:

(a) the total number of parameters in the base model,
(b) the number of trainable LoRA parameters under the default configuration,
(c) the percentage of parameters being trained.

Briefly explain why this percentage is small and how LoRA achieves this reduction.

#### Task 2.2 Training with Increasing Data

The GSM8K training split contains 7,473 question-answer pairs. Rather than starting at full scale, begin with a smaller run to build intuition, validate your setup, and measure the first performance jump.

**QUESTION 5: (25 points)** Train a LoRA SFT model using 1,000 training examples. Evaluate on 100 GSM8K test questions and report the accuracy. Include a brief comment on whether the improvement over the baseline matches your expectations.

Scaling up training data improves performance, but it also increases runtime and cost. Before running larger experiments, it is worth forming a hypothesis about whether scaling will meaningfully help.

**QUESTION 6: (10 points)** Hypothesis question (write before running larger training): Do you think scaling from 1,000 examples to 3,000 and/or all 7,473 examples is worth the additional compute? What do you expect the accuracy gains to look like (roughly), and why? In your answer, discuss whether you would scale in multiple steps or jump directly to the full dataset.

**QUESTION 7: (20 points: 5 + 5 + 10)** Now scale up your training data (recommended: 3,000 examples, and optionally the full 7,473). Evaluate each trained model on the same 100-question test subset and report the accuracies.

Finally, plot accuracy as a function of the number of training examples (x-axis: 0, 1000, 3000; y-axis: accuracy). Describe the trend you observe and comment on diminishing returns in data scaling for SFT.

#### Task 2.3 Qualitative Analysis

**QUESTION 8: (10 points)** Compare the base model and your best SFT model on the same 3 failure examples you identified in Task 1. For each example, show both models' responses side by side. Does the SFT model fix any of these errors?

**QUESTION 9: (5 points)** Identify 2 examples where your best SFT model still fails. What types of errors persist after fine-tuning? Does the model struggle more with arithmetic, multi-step reasoning, or problem comprehension?

Note: You should expect to reach around 45-50% accuracy by this time.

---

### Task 3 — Few-Shot Prompting

Fine-tuning changes the model's weights. An orthogonal approach is to change the prompt: by including a small number of worked examples ("shots") before the actual question, we can steer the model toward a desired reasoning style and a more consistent output format at inference time, without updating any parameters.

#### Task 3.1 Few-Shot Setup

In k-shot prompting, we prepend k complete (question, solution) pairs as user/assistant turns in the chat before the actual test question. Conceptually, the model sees:

```
[system] System prompt
[user]   Example question 1
[assistant] Example solution 1
[user]   Example question 2
[assistant] Example solution 2
...
[user]   Actual test question
[assistant] <model generates here>
```

For fairness, use a **fixed pool** of k demonstration examples for all models you evaluate (base, SFT). The goal is to keep the prompt content constant so that differences in performance reflect differences in the model rather than differences in the examples.

Practical note: The quality of your demonstrations matters. A natural question to ask is: what is the best source for high-quality step-by-step reasoning answers to use as demonstrations? Yes, that is correct — it is the place to go now. Any guesses?

#### Task 3.2 Few-Shot Experiments

**QUESTION 10: (20 points)** Evaluate k-shot prompting (use k = 3) on:

1. the base model,
2. your LoRA SFT model trained on 3k examples,

Report the k-shot results alongside the corresponding no-demonstration baseline results, and compute the improvement (Δ).

**QUESTION 11: (15 points)** Analyze the effect of few-shot prompting on each model:

- Does few-shot help the base model? If not, why might it perform worse with demonstrations?
- Does few-shot help the SFT models? By how much?
- Which model benefits the most from few-shot prompting, and why?

---

### Task 4 — Beyond Scaling: Quality Matters (Open-Ended Exploration)

By now, you should have improved substantially over the base model using LoRA SFT and few-shot prompting. However, you may also have noticed a recurring pattern: scaling (more training examples, longer prompts, more shots) helps, but the gains can slow down. At some point, the bottleneck is less about how much data you have and more about how good the supervision is.

This sets up the central idea for the final part of the project: **improving reasoning performance is often a data-quality problem.** If your training solutions are noisy, terse, poorly structured, or inconsistent, then adding more of them may not teach the model the behavior you want. In contrast, training on fewer but higher-quality, well-structured solutions can lead to surprisingly large improvements.

Using a quality-focused approach that is feasible within the scope of this course, we were able to obtain a model in the **low-to-mid 70% accuracy range** on this evaluation setup. This number serves as a practical reference point.

**QUESTION 12: (10 points)** Qualitative reflection (short): Based on your results so far, what do you think is limiting performance? For each of the following, justify with 2–4 concrete observations from your own failures:

- arithmetic reliability,
- multi-step planning / long-horizon reasoning,
- problem comprehension,
- output consistency / extraction failures,
- training data quality (solution style, structure, or noise).

---

### Task 5 — Open Challenge: Push Toward the Ceiling

Your goal is to beat your own best score from Tasks 1–3. This final task is intentionally exploratory: you will propose a hypothesis, run a small set of targeted experiments, and report what worked (or did not work) and why.

**Grading emphasis:** This task rewards scientific reasoning more than brute-force compute.

- **25 points:** quality of hypotheses, experimental design, and analysis (clear ablations, controlled variables, honest interpretation).
- **10 points:** final accuracy improvement (on the same 100-question evaluation subset).
- **Bonus (+10):** if you surpass the staff reference number, you receive a bonus on top of the project score.

Below are sensible directions to explore (choose one or combine a small number of them):

- **Demonstration quality:** curate a small set of high-quality few-shot demonstrations (clear steps, correct arithmetic, consistent final-answer marker) and measure gains over your current few-shot setup.
- **Prompt and format engineering:** redesign the system prompt and/or answer marker to reduce extraction failures and encourage disciplined reasoning; test a small set of prompt variants and lock one in.
- **Targeted training data selection:** train on a subset of GSM8K that emphasizes harder multi-step problems or known failure modes of the base model.
- **Self-consistency at inference:** sample multiple solutions (temperature > 0) and take a majority vote over extracted answers.
- **Light hyperparameter tuning:** adjust LoRA rank, learning rate, epochs, or target modules; justify expected effects and report what changed.
- **Quality-focused supervision:** generate or obtain a set of higher-quality step-by-step solutions and fine-tune on them (filtering for correctness when possible).

**QUESTION 13:** Design and implement your favourite strategy to improve upon your best score. In your report, include:

(a) **Hypothesis:** what you think will help and why.
(b) **Method:** what you changed (prompting, training data, hyperparameters, inference procedure, etc.).
(c) **Results:** accuracy on the same 100-question evaluation subset (include your baseline for comparison).
(d) **Analysis:** what you learned from the experiment(s), including at least one failure mode or unexpected result.

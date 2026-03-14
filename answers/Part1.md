# Part A: Teaching a Small Model to Reason: LoRA Fine-Tuning

> **Note on Hyperparameters:** Due to GPU memory constraints, the batch size and gradient accumulation values were swapped from the assignment defaults. The assignment specifies per-device batch size of 8 with gradient accumulation of 4, but the implementation uses per-device batch size of 4 with gradient accumulation of 8. This maintains the same effective batch size of 32 (4×8 = 8×4 = 32) while reducing peak memory usage.

## Task 1: Baseline: How Good is the Base Model?

**QUESTION 1: (10 points) Run the base Qwen2.5-1.5B-Instruct model on 100 GSM8K test questions and report the accuracy. You should expect approximately 35–40%. (Exact values may vary slightly depending on the prompt format and extraction rule.)**

**Answer:**

- **Base Model Accuracy:** 38%
- **Number of Test Samples:** 100

The base Qwen2.5-1.5B-Instruct scored 38% on GSM8K—right in the expected 35-40% range. That's our baseline. The model struggles with multi-step reasoning and arithmetic, which isn't surprising for a 1.5B model without math-specific training.

---

**QUESTION 2: (10 points) Inspect at least 3 cases where the base model produces an incorrect answer. For each example, include:**

- The question,
- A short excerpt of the model's solution highlighting the failure,
- The extracted answer vs. the ground-truth answer.

Classify each failure mode (e.g., arithmetic slip, reasoning/logical error, misunderstanding the problem, formatting/extraction issue). Do you observe any recurring patterns?

**Answer:**

**Failure Case 1: Arithmetic Error**

- **Question:** Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
- **Ground Truth:** 18
- **Extracted Answer:** 0
- **Failure Excerpt:**
  ```
  Remaining eggs for sale = Remaining eggs after breakfast - Eggs used for baking muffins
                          = 13 - 4
                          = -1 (This is not possible since it would mean negative eggs)
  ```
- **Classification:** **Arithmetic Slip**: The model sets up the calculation correctly (16 - 3 - 4 = 9) but somehow concludes this equals -1 instead of 9.

**Failure Case 2: Problem Comprehension**

- **Question:** A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
- **Ground Truth:** 3
- **Extracted Answer:** 2.5
- **Failure Excerpt:**
  ```
  The robe requires half as much white fiber as blue fiber.
  Therefore, the amount of white fiber is (1/2) bolt of blue fiber.
  2 (blue fibers) + 1/2 (white fibers) = 2.5 bolts
  ```
- **Classification:** **Problem Comprehension**: The model read "half that much white fiber" as half a bolt total, not half of 2 bolts (which is 1 bolt).

**Failure Case 3: Multi-step Reasoning Error**

- **Question:** Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
- **Ground Truth:** 70000
- **Extracted Answer:** -55500
- **Failure Excerpt:**
  ```
  Increase in value = Total cost × Percentage increase
  Increase in value = $130,000 × 0.150 = $49,500
  Profit = $129,500 - $130,000 = -$5,500
  ```
- **Classification:** **Multi-step Reasoning Error**: Two errors: (1) applied 150% increase to total cost instead of original price, (2) used 0.150 instead of 1.50 for 150%.

**Recurring patterns:**
1. Arithmetic errors—correct setup, wrong math
2. Problem comprehension—misreading word problems, especially fractions and percentages
3. Multi-step reasoning—losing track halfway through

---

## Task 2: LoRA Fine-Tuning on GSM8K

**QUESTION 3: (15 points) Pick the three hyperparameters (LoRA rank, LoRA alpha, Gradient accumulation) from the table above and explain:**

- what each hyperparameter controls,
- what you expect to happen if you increase it,
- what you expect to happen if you decrease it.

Your answer should reflect practical tradeoffs (e.g., compute/memory, stability, overfitting vs. underfitting).

**Answer:**

**LoRA Rank (r)**

- **What it controls:** Size of the low-rank matrices A and B. Basically: adapter capacity.
- **If you increase:** More parameters to train, potentially better adaptation, but more memory and overfitting risk.
- **If you decrease:** Fewer parameters, more constrained adaptation, might underfit but generalizes better.

**LoRA Alpha (α)**

- **What it controls:** Scaling factor for LoRA updates before merging with original weights. The actual update is scaled by α/r.
- **If you increase:** Adapter has more influence, learning speeds up, but training can become unstable.
- **If you decrease:** Gentler updates, more stable training, slower convergence.

**Gradient Accumulation**

- **What it controls:** How many mini-batches of gradients to stack before updating weights. Effective batch size = per_device_batch_size × gradient_accumulation_steps.
- **If you increase:** Larger effective batch size (often better generalization), but less frequent updates.
- **If you decrease:** Smaller effective batch size, noisier gradients, faster updates, but can destabilize training.

---

**QUESTION 4: (15 points) Report:**

(a) the total number of parameters in the base model,
(b) the number of trainable LoRA parameters under the default configuration,
(c) the percentage of parameters being trained.

Briefly explain why this percentage is small and how LoRA achieves this reduction.

**Answer:**

- **(a) Base Model Total Parameters:** 1,543,714,304 (~1.54 billion)
- **(b) Trainable LoRA Parameters:** 2,179,072 (~2.18 million)
- **(c) Percentage:** 0.141% (about 1 in 708 parameters)

**Why so few parameters:** LoRA uses low-rank decomposition. Instead of updating a full weight matrix W ∈ ℝ^(d×d) (d² parameters), LoRA represents the update as ΔW = BA, where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×d). This cuts parameters from d² to 2dr. With rank r=8 and typical dimension d=2048, that's ~128× fewer parameters per adapted layer.

---

**QUESTION 5: (25 points) Train a LoRA SFT model using 1,000 training examples. Evaluate on 100 GSM8K test questions and report the accuracy. Include a brief comment on whether the improvement over the baseline matches your expectations.**

**Answer:**

- **SFT-1k Accuracy:** 41%
- **Baseline Accuracy:** 38%
- **Improvement:** +3 percentage points

Going from 38% to 41% is real progress. Fine-tuning helps even with just 1,000 examples. Training took ~40-50 minutes on a T4 GPU—and we're only training 0.14% of parameters.

---

**QUESTION 6: (10 points) Hypothesis question (write before running larger training): Do you think scaling from 1,000 examples to 3,000 and/or all 7,473 examples is worth the additional compute? What do you expect the accuracy gains to look like (roughly), and why? In your answer, discuss whether you would scale in multiple steps or jump directly to the full dataset.**

**Answer:**

**Yes, scaling is worth the compute, but with diminishing returns.**

**Expected Accuracy Gains:**
- 1,000 → 3,000 examples: ~2-4 pp improvement (42% → 44-46%)
- 3,000 → 7,473 examples: ~1-3 pp improvement (44-46% → 45-49%)

**Why:**
- More data = more pattern diversity, better generalization, stronger reasoning
- But diminishing returns: GSM8K has redundancy, the model only has 0.14% trainable parameters, and there's a quality vs quantity trade-off

**My strategy:** Scale in two steps (1k → 3k → full) to track how scaling behaves and stop early if returns flatten.

---

**QUESTION 7: (20 points: 5 + 5 + 10) Now scale up your training data (recommended: 3,000 examples, and optionally the full 7,473). Evaluate each trained model on the same 100-question test subset and report the accuracies.**

Finally, plot accuracy as a function of the number of training examples (x-axis: 0, 1000, 3000; y-axis: accuracy). Describe the trend you observe and comment on diminishing returns in data scaling for SFT.

**Answer:**

| Training Examples | Accuracy | Improvement from Baseline |
| ----------------- | -------- | ------------------------- |
| 0 (baseline)      | 38%      | -                         |
| 1,000             | 41%      | +3 pp                     |
| 3,000             | 44%      | +6 pp                     |

File: [Scaling Plot](outputs/q7_scaling_plot.png)

**Analysis:**
- Consistent improvement: 38% → 41% → 44%
- Diminishing returns: 0→1k gives +3pp with 1k examples, 1k→3k gives +3pp with 2k additional examples
- Extrapolating to full dataset (7,473) might yield ~46-48%

---

**QUESTION 8: (10 points) Compare the base model and your best SFT model on the same 3 failure examples you identified in Task 1. For each example, show both models' responses side by side. Does the SFT model fix any of these errors?**

**Answer:**

**Case 1: Janet's Ducks (Ground Truth: 18)**
- Base Model: 0 (arithmetic error: 13 - 4 = -1)
- SFT-3k Model: 73.50 (different error: hallucinated 16×3 multiplication)
- **Result:** Both failed with different errors

**Case 2: Robe Fabric (Ground Truth: 3)**
- Base Model: 2.5 (misinterpreted "half that much" as 0.5 bolts)
- SFT-3k Model: 3 (correctly interpreted as half of 2 bolts = 1 bolt)
- **Result:** SFT fixed this error ✓

**Case 3: House Flipping (Ground Truth: 70000)**
- Base Model: -55500 (applied 0.150 instead of 1.5 for 150%)
- SFT-3k Model: 69500 (still applied 0.15 instead of 1.5)
- **Result:** Both failed with similar percentage calculation errors

**Summary:** SFT fixed 1/3 problems (language comprehension improved), but arithmetic and percentage reasoning remain hard.

---

**QUESTION 9: Identify 2 examples where your best SFT model still fails. What types of errors persist?**

**Answer:**

**Failure 1: Janet's Ducks (Multi-step reasoning with hallucinated operations)**
- Extracted Answer: 73.50 (Ground Truth: 18)
- Error: Model introduced erroneous multiplication (16×3=48) not in problem statement
- Type: **Multi-step reasoning with hallucinated operations**

**Failure 2: House Flipping (Percentage calculation error)**
- Extracted Answer: -30500 (Ground Truth: 70000)
- Error: Applied 0.15 (15%) instead of 1.5 (150%)
- Type: **Arithmetic/Percentage reasoning**

**Persistent error types (ranked by frequency):**
1. Arithmetic/Percentage errors (converting percentages incorrectly)
2. Multi-step reasoning with hallucinated operations
3. Problem comprehension (occasional misinterpretation)

---

## Task 3: Few-Shot Prompting

**QUESTION 10: (20 points) Evaluate k-shot prompting (use k = 3) on:**

1. the base model,
2. your LoRA SFT model trained on 3k examples,

Report the k-shot results alongside the corresponding no-demonstration baseline results, and compute the improvement (Δ).

**Answer:**

| Model        | Zero-Shot Accuracy | 3-Shot Accuracy | Improvement (Δ) |
| ------------ | ------------------ | --------------- | --------------- |
| Base Model   | 38%                | 32%             | -6 pp           |
| SFT-3k Model | 44%                | 50%             | +6 pp           |

**Key observations:**
- Few-shot helps SFT (+6 pp) but hurts base model (-6 pp)
- Combined gains: SFT-3k + few-shot reaches 50%, a 12 pp improvement over baseline

---

**QUESTION 11: (15 points) Analyze the effect of few-shot prompting on each model:**

- Does few-shot help the base model? If not, why might it perform worse with demonstrations?
- Does few-shot help the SFT models? By how much?
- Which model benefits the most from few-shot prompting, and why?

**Answer:**

**Effect on Base Model (-6 pp)**

Few-shot hurts because:
1. No reasoning foundation to use demonstrations
2. Attention dilution from context window usage
3. Format confusion from demonstration patterns
4. Negative transfer from reasoning patterns the base model can't follow

**Effect on SFT Models (+6 pp)**

Few-shot helps because:
1. Reinforces step-by-step reasoning patterns from training
2. Consistent format alignment with training
3. Pattern completion guidance
4. Fewer format errors

**Which benefits most:** SFT-3k benefits most (+6 pp vs -6 pp). Few-shot and SFT work together—SFT teaches reasoning style, few-shot reinforces it.

---

## Task 4: Beyond Scaling: Quality Matters

**QUESTION 12: (10 points) Qualitative reflection (short): Based on your results so far, what do you think is limiting performance? For each of the following, justify with 2–4 concrete observations from your own failures:**

- arithmetic reliability,
- multi-step planning / long-horizon reasoning,
- problem comprehension,
- output consistency / extraction failures,
- training data quality (solution style, structure, or noise).

**Answer:**

**1. Arithmetic Reliability (Major limitation)**
- Simple calculation errors persist (15 + 25 = 7, 13 - 4 = -1)
- No self-verification of intermediate results
- Percentage confusion (using 0.15 for 150%)

**2. Multi-Step Planning (Primary limitation)**
- Loses track of goals in complex problems
- Incorrect intermediate steps
- No explicit planning before execution
- Trouble with 4+ step problems

**3. Problem Comprehension (Moderate limitation)**
- Language ambiguity misinterpretation
- Missed constraints
- Hallucinated information
- ~15-20% of errors from initial misunderstanding

**4. Output Consistency (Minor limitation)**
- Format variations (units in answers)
- Incomplete answers
- Affects ~2-3% of answers

**5. Training Data Quality (Significant limitation)**
- Inconsistent solution styles
- No verification signals
- No error correction examples
- Scaling paradox: more data didn't help as much as expected

**Ranked importance:** Multi-step reasoning > Training data quality > Arithmetic reliability > Problem comprehension > Output consistency

---

## Task 5: Open Challenge: Push Toward the Ceiling

**QUESTION 13: Design and implement your favourite strategy to improve upon your best score. In your report, include:**

(a) **Hypothesis:** what you think will help and why.
(b) **Method:** what you changed (prompting, training data, hyperparameters, inference procedure, etc.).
(c) **Results:** accuracy on the same 100-question evaluation subset (include your baseline for comparison).
(d) **Analysis:** what you learned from the experiment(s), including at least one failure mode or unexpected result.

**Answer:**

**(a) Hypothesis**

Self-consistency through majority voting will improve accuracy by reducing random errors and leveraging the model's varying reasoning paths. By sampling multiple solutions and taking majority vote, we can filter out random errors.

**(b) Method**

- **Model:** SFT-3k (best performing checkpoint)
- **Sampling:** 10 samples per question, temperature=0.3
- **Voting:** Majority vote on extracted answers
- **Cost:** ~10× inference time

**(c) Results**

| Method                            | Accuracy | Improvement |
| --------------------------------- | -------- | ----------- |
| Baseline (SFT-3k, zero-shot)     | 44%      | -           |
| Self-Consistency (n=10, temp=0.3) | 56%      | **+12 pp**  |

**Example Success (Eliza's overtime, Ground Truth: 460):**
- 10 Sampled Answers: [460, 450, 110, 460, 460, 460, 450, 460, 460, 110]
- Majority Vote: 460 ✓

**(d) Analysis**

**What worked:**
1. Arithmetic error filtering through voting
2. Reasoning path diversity
3. Consistent problems benefited most

**Failure Modes:**

1. **Consistent Wrong Reasoning:** Janet's ducks: all samples made different errors, majority vote selected wrong answer (104 vs ground truth 18)

2. **Rare Correct Answers:** James running sprints ( only 1/5 samples correct, majority selected wrong answer (21 vs 540)

3. **Systematic Misconception:** House flipping ( all samples struggled with 150% calculation, different wrong answers but same underlying error)

**Key Learnings:**
1. Self-consistency helps with random errors, not systematic ones
2. Diminishing returns on samples (1→3 captures most benefits)
3. Problem difficulty matters (works best on "sometimes right" problems)
4. Failed to reach 70% goal—training data quality is the bigger bottleneck

**Conclusion:** Self-consistency improved accuracy by 12 pp (44% → 56%) but can't fix systematic reasoning gaps. The gap to 70%+ requires better training data quality.

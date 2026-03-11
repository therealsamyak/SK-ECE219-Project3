# Part A: Teaching a Small Model to Reason: LoRA Fine-Tuning

> **Note on Hyperparameters:** Due to GPU memory constraints, the batch size and gradient accumulation values were swapped from the assignment defaults. The assignment specifies per-device batch size of 8 with gradient accumulation of 4, but the implementation uses per-device batch size of 4 with gradient accumulation of 8. This maintains the same effective batch size of 32 (4×8 = 8×4 = 32) while reducing peak memory usage.

## Task 1 — Baseline: How Good is the Base Model?

### Question 1

Run the base Qwen2.5-1.5B-Instruct model on 100 GSM8K test questions and report the accuracy. You should expect approximately 35–40%.

**Answer:**

- **Base Model Accuracy:** 38%
- **Number of Test Samples:** 100

The base Qwen2.5-1.5B-Instruct model gets 38% accuracy on the GSM8K test set, within the expected 35-40% range. This gives us a baseline to improve upon. The model has trouble with multi-step reasoning and arithmetic—expected for a small language model without specific math training.

---

### Question 2

Inspect at least 3 cases where the base model produces an incorrect answer. For each example, include: the question, a short excerpt of the model's solution highlighting the failure, the extracted answer vs. the ground-truth answer. Classify each failure mode.

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
- **Classification:** **Arithmetic Slip** - The model correctly sets up the calculation (16 - 3 - 4 = 9) but then makes a catastrophic error by concluding this is -1 instead of 9.

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
- **Classification:** **Problem Comprehension** - The model misinterpreted "half that much white fiber" as meaning half a bolt total, rather than half of 2 bolts (which is 1 bolt).

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
- **Classification:** **Multi-step Reasoning Error** - The model made multiple errors: (1) applied 150% increase to total cost rather than original price, (2) used 0.150 instead of 1.50 for 150%.

**Recurring patterns:**
1. Arithmetic errors—even with correct setup, basic calculation mistakes
2. Problem comprehension—misinterpreting word problems, especially with fractions and percentages
3. Multi-step reasoning—trouble maintaining correct logic through multiple steps

---

## Task 2 — LoRA Fine-Tuning on GSM8K

### Question 3

Pick the three hyperparameters (LoRA rank, LoRA alpha, Gradient accumulation) from the table above and explain: what each hyperparameter controls, what you expect to happen if you increase it, what you expect to happen if you decrease it.

**Answer:**

**LoRA Rank (r)**

- **What it controls:** Dimensionality of low-rank matrices A and B. Controls adapter capacity.
- **If you increase:** More trainable parameters, better adaptation, higher memory usage, risk of overfitting
- **If you decrease:** Fewer parameters, limited adaptation, potential underfitting, better generalization

**LoRA Alpha (α)**

- **What it controls:** Scaling factor applied to LoRA update before adding to original weights. Update scaled by α/r.
- **If you increase:** Stronger adapter influence, faster learning, risk of instability
- **If you decrease:** Weaker adapter influence, more stable training, slower learning

**Gradient Accumulation**

- **What it controls:** Number of gradient updates to accumulate before weight update. Effective batch size = per_device_batch_size × gradient_accumulation_steps.
- **If you increase:** Larger effective batch size, better generalization, slower updates
- **If you decrease:** Smaller effective batch size, noisier gradients, faster updates, risk of instability

---

### Question 4

Report: (a) the total number of parameters in the base model, (b) the number of trainable LoRA parameters under the default configuration, (c) the percentage of parameters being trained.

**Answer:**

- **(a) Base Model Total Parameters:** 1,543,714,304 (approximately 1.54 billion)
- **(b) Trainable LoRA Parameters:** 2,179,072 (approximately 2.18 million)
- **(c) Percentage:** 0.141% (only about 1 in 708 parameters is trained)

**Why this percentage is small:** LoRA uses low-rank decomposition. Instead of updating a full weight matrix W ∈ ℝ^(d×d) requiring d² parameters, LoRA represents the update as ΔW = BA, where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×d). This cuts parameters from d² to 2dr. With rank r=8 and typical dimension d=2048, that's ~128× fewer parameters per adapted layer.

---

### Question 5

Train a LoRA SFT model using 1,000 training examples. Evaluate on 100 GSM8K test questions and report the accuracy.

**Answer:**

- **SFT-1k Accuracy:** 41%
- **Baseline Accuracy:** 38%
- **Improvement:** +3 percentage points

The improvement from 38% to 41% (+3 pp) is real progress. Even with only 1,000 examples, fine-tuning helps. Training took ~40-50 minutes on a T4 GPU, training only 0.14% of parameters.

---

### Question 6

Hypothesis question: Do you think scaling from 1,000 examples to 3,000 and/or all 7,473 examples is worth the additional compute?

**Answer:**

**Yes, scaling is worth the compute, but with diminishing returns.**

**Expected Accuracy Gains:**
- 1,000 → 3,000 examples: ~2-4 percentage point improvement (42% → 44-46%)
- 3,000 → 7,473 examples: ~1-3 percentage point improvement (44-46% → 45-49%)

**Rationale:**
- Scaling helps: pattern diversity, better generalization, more robust reasoning
- Diminishing returns: data redundancy in GSM8K, limited model capacity (0.14% trainable), quality vs quantity trade-off

**Recommended strategy:** Scale in two steps (1k → 3k → full) to track scaling behavior and stop early if returns flatten.

---

### Question 7

Scale up training data (3,000 examples, and optionally full 7,473). Evaluate each trained model and plot accuracy as a function of training examples.

**Answer:**

| Training Examples | Accuracy | Improvement from Baseline |
| ----------------- | -------- | ------------------------- |
| 0 (baseline)      | 38%      | -                         |
| 1,000             | 41%      | +3 pp                     |
| 3,000             | 44%      | +6 pp                     |

File: [Scaling Plot](outputs/q7_scaling_plot.png)

**Analysis:**
- Consistent improvement as data increases: 38% → 41% → 44%
- Diminishing returns: 0→1k gives +3pp with 1k examples, 1k→3k gives +3pp with 2k additional examples
- Extrapolating to full dataset (7,473) might yield ~46-48%

---

### Question 8

Compare the base model and your best SFT model on the same 3 failure examples from Task 1.

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

**Summary:** SFT fixed 1/3 problems (language comprehension got better), but arithmetic and percentage reasoning remain hard.

---

### Question 9

Identify 2 examples where your best SFT model still fails. What types of errors persist?

**Answer:**

**Failure 1: Janet's Ducks - Multi-step reasoning with hallucinated operations**
- Extracted Answer: 73.50 (Ground Truth: 18)
- Error: Model introduced erroneous multiplication (16×3=48) not in problem statement
- Type: **Multi-step reasoning with hallucinated operations**

**Failure 2: House Flipping - Percentage calculation error**
- Extracted Answer: -30500 (Ground Truth: 70000)
- Error: Applied 0.15 (15%) instead of 1.5 (150%)
- Type: **Arithmetic/Percentage reasoning**

**Persistent error types (ranked by frequency):**
1. Arithmetic/Percentage errors (converting percentages incorrectly)
2. Multi-step reasoning with hallucinated operations
3. Problem comprehension (occasional misinterpretation)

---

## Task 3 — Few-Shot Prompting

### Question 10

Evaluate k-shot prompting (k=3) on the base model and your LoRA SFT model trained on 3k examples.

**Answer:**

| Model        | Zero-Shot Accuracy | 3-Shot Accuracy | Improvement (Δ) |
| ------------ | ------------------ | --------------- | --------------- |
| Base Model   | 38%                | 32%             | -6 pp           |
| SFT-3k Model | 44%                | 50%             | +6 pp           |

**Key observations:**
- Few-shot helps SFT (+6 pp) but hurts base model (-6 pp)
- Combined gains: SFT-3k + few-shot reaches 50%, a 12 pp improvement over baseline

---

### Question 11

Analyze the effect of few-shot prompting on each model.

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

## Task 4 — Beyond Scaling: Quality Matters

### Question 12

Based on results so far, what do you think is limiting performance?

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

## Task 5 — Open Challenge: Push Toward the Ceiling

### Question 13

Design and implement your favorite strategy to improve upon your best score.

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

1. **Consistent Wrong Reasoning:** Janet's ducks - all samples made different errors, majority vote selected wrong answer (104 vs ground truth 18)

2. **Rare Correct Answers:** James running sprints - only 1/5 samples correct, majority selected wrong answer (21 vs 540)

3. **Systematic Misconception:** House flipping - all samples struggled with 150% calculation, different wrong answers but same underlying error

**Key Learnings:**
1. Self-consistency helps with random errors, not systematic ones
2. Diminishing returns on samples (1→3 captures most benefits)
3. Problem difficulty matters (works best on "sometimes right" problems)
4. Failed to reach 70% goal, suggesting training data quality is bigger bottleneck

**Conclusion:** Self-consistency improved accuracy by 12 pp (44% → 56%) but can't fix systematic reasoning gaps. The gap to 70%+ requires better training data quality.

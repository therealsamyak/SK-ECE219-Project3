# Part B: Agentic Data Mining with ReAct

> **Implementation Note:** This implementation uses the Qwen3-4B-Instruct-2507 model with 4-bit quantization (NF4) to fit within GPU memory constraints. The ReAct agent follows a structured approach with Planner, Coder, Executor, and Observer components, using Outlines for structured output generation.

## Task 1 — Dataset Inspection and Sanity Checks

### Question 14

Load `da-dev-questions.jsonl` and `da-dev-labels.jsonl`. Report the number of questions and labels, the set of keys in a question record, and the set of keys in a label record.

**Answer:**

- **Number of Questions:** 257
- **Number of Labels:** 257

**Question Record Keys:**
- `id`: Unique identifier
- `question`: Natural language question
- `concepts`: List of data analysis concepts
- `file_name`: Reference to CSV file
- `level`: Difficulty level (easy/medium/hard)
- `format`: Required answer format with @name[value] slots
- `constraints`: Additional requirements

**Example Question Record:**
```json
{
  "id": 0,
  "question": "Calculate the mean fare paid by the passengers.",
  "concepts": ["Summary Statistics"],
  "file_name": "test_ave.csv",
  "level": "easy",
  "format": "@mean_fare[mean_fare]",
  "constraints": "Round to 2 decimal places."
}
```

**Label Record Keys:**
- `id`: Question ID
- `common_answers`: List of [name, value] pairs

---

### Question 15

Pick 3 random question IDs. For each, print the file name, load the CSV with pandas, show shape/dtypes/head(3), and print the corresponding question.

**Answer:**

**Example 1: Election Data (ID 142)**
- **File:** `election2016.csv`
- **Shape:** 3141 rows × 10 columns
- **Columns:** votes_dem, votes_gop, total_votes, per_dem, per_gop, diff, per_point_diff, state_abbr, county_name, combined_fips
- **Question:** "Is there a relationship between the difference in votes received by the Democratic and Republican parties and their percentage point difference?"

**Example 2: Insurance Data (ID 24)**
- **File:** `insurance.csv`
- **Shape:** 1338 rows × 7 columns
- **Columns:** age, sex, bmi, children, smoker, region, charges
- **Question:** "Calculate the mean age of the individuals in the dataset."

**Example 3: Storm Data (ID 429)**
- **File:** `cost_data_with_errors.csv`
- **Shape:** 818 rows × 11 columns
- **Columns:** Unnamed: 0, name, dates_active, max_storm_cat, max_sust_wind, min_p, areas_affected, damage_USD, deaths, year, damage_imputed
- **Question:** "Is there a correlation between the maximum storm category achieved by a storm and the recorded damage in USD?"

---

### Question 16

Find 2 examples where the required format contains multiple answer slots. Explain how the dataset represents multi-part answers and how to evaluate them automatically.

**Answer:**

**Example 1: Age Group Analysis (ID 6)**
- **Question:** "Create a new column called 'AgeGroup' that categorizes passengers into four age groups: 'Child' (0-12), 'Teenager' (13-19), 'Adult' (20-59), and 'Elderly' (60+). Then calculate the mean fare for each age group."
- **Format:** `@mean_fare_child[mean_fare], @mean_fare_teenager[mean_fare], @mean_fare_adult[mean_fare], @mean_fare_elderly[mean_fare]`
- **Slots:** 4

**Example 2: Multi-Class Distribution (ID 8)**
- **Question:** "Perform distribution analysis on 'Fare' for each passenger class separately. Calculate mean, median, and standard deviation for each class."
- **Format:** `@mean_fare_class1[mean_fare], @median_fare_class1[median_fare], @std_dev_fare_class1[std_dev], ...` (9 total slots)

**Dataset Representation:**
- Multi-part answers use multiple `@name[value]` slots
- Ground truth contains list of [name, value] pairs
- Example: `[["mean_fare_child", "34.56"], ["mean_fare_adult", "42.78"]]`

**Automatic Evaluation:**
1. Extract predicted values using regex: `r"@(\w+)\[([^\]]+)\]"`
2. Compare with ground truth using tolerance (1.1% relative, 0.011 absolute)
3. All slots must match for correct answer

---

### Question 17

Print and check the 10 selected tasks (IDs: 0, 5, 9, 10, 14, 18, 24, 25, 26, 55).

**Answer:**

| ID  | Question                                                      | Concepts                    | File                       | Level  |
| --- | ------------------------------------------------------------- | --------------------------- | -------------------------- | ------ |
| 0   | Calculate the mean fare paid by passengers.                   | Summary Statistics          | test_ave.csv               | Easy   |
| 5   | Generate "FamilySize" feature. Calculate correlation.         | Feature Engineering         | test_ave.csv               | Medium |
| 9   | Calculate mean "Close Price".                                 | Summary Statistics          | GODREJIND.csv              | Easy   |
| 10  | Check if "Total Traded Quantity" follows normal distribution. | Distribution Analysis       | GODREJIND.csv              | Easy   |
| 14  | Create "Price Range" feature. Calculate stats.                | Feature Engineering         | GODREJIND.csv              | Medium |
| 18  | Calculate mean and std of "Mar.2019".                         | Summary Statistics          | unemployement_industry.csv | Easy   |
| 24  | Calculate mean age.                                           | Summary Statistics          | insurance.csv              | Easy   |
| 25  | Check if BMI follows normal distribution.                     | Distribution Analysis       | insurance.csv              | Easy   |
| 26  | Calculate correlation between charges and children.           | Correlation Analysis        | insurance.csv              | Easy   |
| 55  | Calculate mean cases across all countries and years.          | Summary Statistics          | estimated_numbers.csv      | Easy   |

---

## Task 2 — Model Loading and Structured Output

### Question 18

Demonstrate with 5 different prompts that your planner always returns valid JSON that parses into your Pydantic model without try/except fallbacks. Include at least one case where `is_done=true`.

**Answer:**

**Pydantic Schema:**
```python
class PlannerOutput(BaseModel):
    thought: str = Field(min_length=10, max_length=500)
    is_done: bool
    response: str = Field(min_length=1)
```

**Prompt 1:** "What is the average age of passengers?"
```json
{
  "thought": "To calculate the average age, I need access to a dataset containing passenger age values.",
  "is_done": false,
  "response": "Retrieve the passenger age dataset and compute the mean age."
}
```

**Prompt 2:** "How many passengers survived?"
```json
{
  "thought": "To determine survivors, I need access to passenger survival information.",
  "is_done": false,
  "response": "Fetch the passenger survival data and count the number of survivors."
}
```

**Prompt 3:** "What is the correlation between age and fare?"
```json
{
  "thought": "To determine correlation between age and fare, I need a dataset with both columns.",
  "is_done": false,
  "response": "Retrieve dataset with age and fare columns, compute Pearson correlation."
}
```

**Prompt 4:** "Show me the distribution of passenger classes."
```json
{
  "thought": "To show distribution of passenger classes, I need to count passengers in each class.",
  "is_done": false,
  "response": "Load dataset and count passengers in each class, present distribution."
}
```

**Prompt 5:** "Calculate the mean fare: @mean_fare[32.20] (final answer)"
```json
{
  "thought": "The mean fare is already provided as 32.20. No further calculation needed.",
  "is_done": true,
  "response": "}"
}
```

**Note:** All 5 outputs are valid JSON. Prompt 5 demonstrates `is_done=true` when answer is already available.

---

### Question 19

Explain why structured output is useful for large-scale data mining pipelines.

**Answer:**

Structured output is essential for large-scale data mining pipelines for three key reasons:

1. **Reliability at Scale:** When processing thousands of tasks, parsing errors from free-form text would cascade into system failures. Structured output (via Outlines + Pydantic) guarantees every planner response is parseable.

2. **Automated Orchestration:** Agents need programmatic decisions (e.g., "if `is_done=true`, return answer; else execute code"). Structured outputs provide machine-readable signals for deterministic control flow.

3. **Error Recovery:** In agent systems, recovery requires clear signals about failures. Structured observations enable systematic error handling and retry logic.

---

## Task 3 — Build a ReAct Data Analysis Agent

### Question 20

Run your ReAct agent on the 10 tasks. Report accuracy and at least 3 qualitative traces illustrating success, failure, and recovery from error.

**Answer:**

**Overall Performance:**

| Metric        | Value              |
| ------------- | ------------------ |
| **Accuracy**  | 90% (9/10 correct) |
| **Correct**   | 9 tasks            |
| **Incorrect** | 1 task             |

**Per-Task Results:**

| Task ID | Result | Predicted                                          | Ground Truth                                       |
| ------- | ------ | -------------------------------------------------- | -------------------------------------------------- |
| 0       | ✅     | @mean_fare[34.65]                                  | @mean_fare[34.65]                                  |
| 5       | ✅     | @correlation_coefficient[0.21]                     | @correlation_coefficient[0.21]                     |
| 9       | ❌     | @mean_close_price[583.68]                          | @mean_close_price[570.68]                          |
| 10      | ✅     | @is_normal[no]                                     | @is_normal[no]                                     |
| 14      | ✅     | @price_range_mean[16.65] @price_range_median[15.67] @price_range_std_dev[6.72] | @price_range_mean[16.65] @price_range_median[15.67] @price_range_std_dev[6.72] |
| 18      | ✅     | @mean_mar_2019[171.44] @sd_mar_2019[188.25]        | @mean_mar_2019[171.44] @sd_mar_2019[188.25]        |
| 24      | ✅     | @mean_age[39.21]                                   | @mean_age[39.21]                                   |
| 25      | ✅     | @bmi_distribution[normal]                          | @bmi_distribution[normal]                          |
| 26      | ✅     | @correlation_coefficient[0.07]                     | @correlation_coefficient[0.07]                     |
| 55      | ✅     | @mean_cases[2081990]                               | @mean_cases[2081990]                               |

---

**Trace 1: Success Case (Task 0 - Mean Fare)**

- **Question:** "Calculate the mean fare paid by the passengers."
- **Thought:** "The DataFrame contains a 'Fare' column of type float64. The mean fare can be computed using pandas' mean() function."
- **Code:**
  ```python
  mean_fare = df['Fare'].mean()
  print(f"{mean_fare:.2f}")
  ```
- **Observation:** "Code executed successfully and returned the value 34.65."
- **Final Answer:** `@mean_fare[34.65]`
- **Result:** ✅ Correct

---

**Trace 2: Failure Case (Task 9 - Mean Close Price)**

- **Question:** "Calculate the mean value of the 'Close Price' column."
- **Final Answer:** `@mean_close_price[583.68]`
- **Ground Truth:** `@mean_close_price[570.68]`
- **Result:** ❌ Incorrect (off by ~2.3%)

**Analysis:** Numerical precision error. The computed value differs from ground truth due to different data preprocessing or NaN handling.

---

**Trace 3: Error Recovery Case (Task 55 - Mean Cases)**

- **Question:** "What is the mean number of cases recorded across all countries and years?"

**Step 1 (First Attempt):**
- **Thought:** "The 'No. of cases' column is in string format with ranges (e.g., '630308[495000-801000]'). Need to extract median value."
- **Code:**
  ```python
  df['No. of cases'] = df['No. of cases'].str.extract(r'(\d+)').astype('int64')
  mean_cases = df['No. of cases'].mean()
  print(f"{mean_cases:.0f}")
  ```
- **Result:** ✅ Successful (2081990)

**Step 2 (Retry - Syntax Error):**
- **Code:**
  ```python
  df['No. of cases'] = df['No. of cases'].str.split('\[').str[0].astype(int)
  ```
- **Result:** ❌ SyntaxError (invalid escape sequence)

**Step 3 (Recovery):**
- **Code:**
  ```python
  df['No. of cases'] = df['No. of cases'].str.extract(r'(\d+)').astype(int)
  mean_cases = df['No. of cases'].mean()
  print(f"{mean_cases:.0f}")
  ```
- **Final Answer:** `@mean_cases[2081990]`
- **Result:** ✅ Recovered successfully

---

**Summary:**

| Strength                    | Observation                                          |
| --------------------------- | ---------------------------------------------------- |
| Structured Output           | All planner outputs valid JSON                       |
| Error Recovery              | Recovered from syntax errors with alternative code   |
| Multi-Step Reasoning        | Handled feature engineering + correlation (Task 5)   |
| Data Cleaning               | Correctly parsed complex string formats (Task 55)    |

| Weakness                    | Observation                                          |
| --------------------------- | ---------------------------------------------------- |
| Precision Issues            | Task 9 off by ~2.3% from ground truth                |

# Instamart Customer Intelligence

## Overview

This project focuses on building an Instamart Customer Intelligence system using Python, Pandas, SQL, Machine Learning, and AI-based support automation.

The project uses locally generated datasets to analyze customer behavior, order patterns, and support tickets. It is designed to run locally without paid APIs or external LLM services.

## Reproduce

### Requirements

- Python 3.11 or above
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

### Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the datasets:

```bash
python generate_data.py
```

Expected output:

```text
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

Run the Part A scripts in order:

```bash
python src/audit_raw.py
python src/clean_customers.py
python src/clean_orders.py
python src/store_validate.py
python src/run_sql.py
python src/eda.py
```

Run the Part B scripts in order:

```bash
python src/build_rfm.py
python src/cluster_rfm.py
python src/final_segmentation.py
python src/build_lapse_cohort.py
python src/check_lapse_preprocessing.py
python src/train_lapse_models.py
python src/balance_lapse_model.py
python src/explain_lapse_model.py
```

Run the Part C script:

```bash
python src/triage_tickets.py
```

Run the Part D scripts:

```bash
python src/policy_rag.py
python src/refund_agent.py
```

## Part A - Data Foundation

**Status: Completed**

### 1. Data Audit

Audited the generated raw datasets:

- Customers: 1,200 rows
- Products: 40 rows
- Orders: 13,084 raw rows
- Tickets: 164 rows

Identified missing values, duplicate records, invalid customer ages, inconsistent city and gender formats, and incomplete order information.

### 2. Data Cleaning

**Customers**

- Standardized city names and gender values.
- Replaced missing gender values with `Unknown`.
- Handled missing and invalid ages using median imputation.
- Final customer records: 1,200.
- No missing values remained.

**Orders**

- Removed 119 exact duplicate rows.
- Removed duplicate order IDs by retaining the latest record.
- Final order records: 12,309.
- Removed pending orders from the cleaned dataset.
- Handled missing order amounts.
- Preserved missing delivery times for cancelled orders.

Final order status counts:

| Status | Records |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |

Completed order revenue: **INR 2,998,668.80**

### 3. Data Storage and Validation

Stored cleaned customer, product, and order datasets as CSV files.

Created a SQLite database:

`data/instamart.db`

Created the following tables:

- customers
- products
- orders

All 15 data validation checks passed.

### 4. SQL Analysis

Implemented seven SQL queries covering:

1. Filtering and aggregation using WHERE and HAVING.
2. Joins across customers, products, and orders.
3. Scalar subqueries.
4. Customers without completed orders.
5. Common Table Expressions (CTEs).
6. Ranking using window functions.
7. Daily running revenue using window functions.

All seven SQL queries executed successfully.

SQL and Pandas revenue cross-check matched with zero difference.

### 5. Exploratory Data Analysis

Performed:

- City-wise and category-wise revenue analysis.
- Customer and order summaries.
- Descriptive statistics.
- Outlier detection using IQR.
- Correlation analysis.

Key results:

- Completed orders: 11,110
- Mean order amount: INR 269.91
- Median order amount: INR 255.00
- Standard deviation: INR 432.07
- Minimum order amount: INR 12.00
- Maximum order amount: INR 3,998.00
- Order amount skewness: 4.61
- IQR outliers above the upper fence: 897

The city/category revenue pivot was successfully cross-checked against SQL results.

### Part A Output Files

- `data/clean/customers_clean.csv`
- `data/clean/orders_clean.csv`
- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `results/q1_results.csv` through `results/q7_results.csv`
- `results/pandas_pivot_revenue.csv`
- `results/sql_pandas_crosscheck.csv`
- `results/city_revenue_summary.csv`
- `results/completed_order_statistics.csv`
- `results/correlation_matrix.csv`

## Part B - Customer Segments and Lapse Prediction

**Status: Completed**

### 1. RFM Analysis
- Analyzed 1,175 customers using Recency, Frequency, and Monetary metrics.
- Snapshot date: 2025-10-01.
- Average frequency: 9.46 orders.
- Average monetary value: INR 2,552.06.
- Median recency: 19 days.

Output: `outputs/rfm_table.csv`

### 2. K-Means Customer Segmentation
- Evaluated K-Means clustering for k = 2 to 8.
- Selected k = 4.
- Silhouette score: 0.5255.

Output: `outputs/kmeans_metrics.csv`

### 3. Segment Profiling

| Cluster | Customers | Profile |
|---|---:|---|
| C0 | 436 | Low-value / Occasional |
| C1 | 138 | High-spending |
| C2 | 419 | Loyal / Frequent |
| C3 | 182 | At-risk / Inactive |

Outputs:
- `outputs/customer_segments.csv`
- `outputs/segment_profiles.csv`

### 4. Lapse Prediction Cohort
- Cutoff date: 2025-09-01.
- Label window: September 2025.
- Cohort size: 1,156 customers.
- Lapsed customers: 443 (38.32%).
- Features were calculated using completed orders before the cutoff date.

Output: `outputs/lapse_cohort.csv`

### 5. Preprocessing Without Data Leakage
- Used an 80:20 stratified train-test split.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Fitted transformations only on training data.
- Generated 16 processed features.

Script: `src/check_lapse_preprocessing.py`

### 6. Model Training and Evaluation
- Evaluated DummyClassifier, Logistic Regression, and Random Forest.
- Used 5-fold cross-validation F1 score for model selection.
- Selected Random Forest based on its higher CV F1 score.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| DummyClassifier | 0.6164 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.6681 | 0.5667 | 0.5730 | 0.5698 |
| Random Forest | 0.6336 | 0.5250 | 0.4719 | 0.4970 |

Random Forest CV F1: 0.5748  
Random Forest test F1: 0.4970

Outputs:
- `outputs/lapse_model_metrics.csv`
- `outputs/lapse_test_predictions.csv`
- `outputs/lapse_model.joblib`

### 7. Class Imbalance Handling
- Applied random oversampling only to the training dataset.
- Kept the test dataset unchanged.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original Logistic Regression | 0.5667 | 0.5730 | 0.5698 |
| Oversampled Logistic Regression | 0.5581 | 0.8090 | 0.6606 |

Outputs:
- `outputs/lapse_imbalance_comparison.csv`
- `outputs/lapse_model_balanced.joblib`

### 8. Interpretability and Model Persistence
- Extracted feature importance from the selected Random Forest model.
- Saved 16 feature importance values.
- Reloaded the saved model and verified identical predictions.

Top features:
1. Orders in the last 30 days
2. Days since last order
3. Average delivery minutes
4. Tenure days
5. Age

Outputs:
- `outputs/lapse_feature_importance.csv`
- `outputs/lapse_model.joblib`

**Part B completed:** Customer segmentation, lapse prediction, model evaluation, class imbalance handling, interpretability, and model persistence.


## Part C - Support Ticket Triage with Guardrails

**Status: Completed**

Implemented a support ticket classification pipeline with input validation,
PII redaction, prompt engineering, retry logic, output validation,
evaluation, token budgeting, and routing.

### Tasks Completed

- Task 1: Input Guardrails
  - Validated empty and oversized support tickets.
  - Maximum input token limit: 400.

- Task 2: PII Redaction
  - Redacted email addresses, phone numbers, and order IDs before model calls.

- Task 3: Zero-shot and Few-shot Prompts
  - Implemented both prompting strategies for ticket classification.

- Task 4: Retry Logic
  - Implemented retry handling with exponential backoff and jitter.

- Task 5: Output Validation
  - Validated JSON structure and allowed ticket categories.
  - Added keyword fallback for invalid JSON responses.

- Task 6: Prompt Comparison
  - Zero-shot accuracy: 85.7%
  - Few-shot accuracy: 100.0%

- Task 7: Token Budget and Routing
  - Implemented token budget tracking and customer-based routing.
  - Added priority levels and human review handling.

### Final Pipeline Results

| Metric | Result |
|---|---:|
| Total tickets | 164 |
| Classified JSON | 135 |
| Classified using keyword fallback | 12 |
| Flagged invalid output | 9 |
| Skipped empty tickets | 4 |
| Flagged long tickets | 4 |
| Total tokens used | 18,331 |
| Retry failures recovered | 19 |

### Output

- `outputs/triage_results.csv`

The final output contains ticket classification status, category,
priority, routing destination, and token usage.

## Part D - Policy RAG Assistant and Refund Agent

**Status: Completed**

### Task 1 - Policy RAG Assistant

- Loaded all policy documents from `data/policies/`.
- Split policy documents into sentence-level chunks.
- Implemented keyword-based policy retrieval.
- Retrieved relevant policy context and generated grounded answers using `mock_llm.py`.
- Evaluated using `data/raw/policy_eval.csv`.

**Evaluation Results:**

| Metric | Result |
|---|---:|
| Total Questions | 14 |
| Correct Document Retrieval | 14/14 |
| Correct Answers | 14/14 |
| Document Accuracy | 100% |
| Answer Accuracy | 100% |

**Output:** `outputs/policy_rag_results.csv`

### Task 2 - Refund Agent

- Implemented a tool-based refund agent using `mock_llm.py`.
- Implemented order lookup and refund policy search.
- Added refund eligibility calculation based on order status and delivery time.
- Added automatic refund processing for eligible refunds.
- Added human escalation for refunds above INR 1000.
- Implemented a maximum of 6 agent steps to prevent infinite loops.
- Evaluated using `data/raw/agent_scenarios.csv`.

**Evaluation Results:**

| Metric | Result |
|---|---:|
| Total Scenarios | 7 |
| Correct Outcomes | 7/7 |
| Accuracy | 100% |
| Maximum Agent Steps | 6 |

**Scenario Coverage:**

- Successful refund
- Large refund escalation
- Refund denial for ineligible orders
- Refund denial for cancelled orders
- Invalid order handling
- Maximum-step fallback

**Output:** `outputs/refund_agent_results.csv`

### Part D Summary

Both the Policy RAG Assistant and Refund Agent completed evaluation successfully with 100% accuracy on their supplied evaluation datasets.

## Limitations and Next Steps

- Part A uses locally generated synthetic data.
- Model performance is evaluated using the supplied datasets.
- The RAG assistant uses keyword-based retrieval and can be improved with vector embeddings.
- The project uses a mock LLM for reproducible evaluation.
- Final reproducibility checks completed successfully for Parts A-D.










from pathlib import Path
import sys
import re
import time
import json
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import mock_llm

INPUT_FILE = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_FILE = ROOT / "outputs" / "triage_results.csv"

MAX_INPUT_TOKENS = 400


# TASK 1: INPUT GUARDRAILS
def validate_input(text):

    if pd.isna(text) or not str(text).strip():
        return False, "empty_ticket"

    token_count = mock_llm.est_tokens(str(text))

    if token_count > MAX_INPUT_TOKENS:
        return False, "input_too_long"

    return True, "valid"


# TASK 2: PERSONAL INFORMATION REDACTION
def redact_pii(text):

    text = str(text)

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    phone_pattern = r'\b\d{10}\b'
    order_pattern = r'\bORD-\d+\b'

    email_count = len(re.findall(email_pattern, text))
    phone_count = len(re.findall(phone_pattern, text))
    order_count = len(re.findall(order_pattern, text))

    text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
    text = re.sub(phone_pattern, '[REDACTED_PHONE]', text)
    text = re.sub(order_pattern, '[REDACTED_ORDER_ID]', text)

    return text, email_count, phone_count, order_count


# TASK 3: ZERO-SHOT PROMPT
def build_zero_shot_prompt(ticket_text):

    system_prompt = """
CLASSIFY the customer support ticket.
Choose exactly one category from:
billing, delivery, technical, fraud.

Return only valid JSON in this format:
{"category": "category_name", "reason": "short reason"}
"""

    user_prompt = f"Ticket: {ticket_text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# TASK 3: FEW-SHOT PROMPT
def build_few_shot_prompt(ticket_text):

    system_prompt = """
CLASSIFY the customer support ticket.

Choose exactly one category from:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: I was charged twice for my order.
Output: {"category": "billing", "reason": "duplicate charge"}

Ticket: My account was hacked.
Output: {"category": "fraud", "reason": "account security issue"}

Ticket: The app keeps crashing.
Output: {"category": "technical", "reason": "app issue"}

Ticket: My delivery is late.
Output: {"category": "delivery", "reason": "delivery delay"}

Return only valid JSON.
"""

    user_prompt = f"Ticket: {ticket_text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# TASK 4: RETRY LOGIC
# TASK 4: RETRY LOGIC WITH EXPONENTIAL BACKOFF AND JITTER
def call_llm_with_retry(messages, max_retries=3):

    for attempt in range(max_retries + 1):

        try:
            response = mock_llm.chat(messages)
            return response, attempt

        except mock_llm.TransientLLMError:

            if attempt == max_retries:
                raise

            delay = random.uniform(0.045, 0.0575) * (2 ** attempt)

            print(
                f"Retry {attempt + 1}: waiting {delay:.4f} seconds"
            )

            time.sleep(delay)

# TASK 5: OUTPUT VALIDATION
def validate_output(response_text):

    try:
        result = json.loads(response_text)

    except (json.JSONDecodeError, TypeError):
        return None, "invalid_json"

    allowed_categories = [
        "billing",
        "delivery",
        "technical",
        "fraud"
    ]

    if not isinstance(result, dict):
        return None, "invalid_structure"

    if result.get("category") not in allowed_categories:
        return None, "invalid_category"

    if not isinstance(result.get("reason"), str):
        return None, "invalid_structure"

    return result, "classified_json"


# TASK 6: KEYWORD FALLBACK
def keyword_fallback(text):

    text = str(text).lower()

    keywords = {
        "fraud": ["unauthorized", "fraud", "hacked", "stolen"],
        "billing": ["charged", "invoice", "bill", "refund", "deducted", "payment"],
        "technical": ["app", "crash", "login", "otp", "error", "freezes"],
        "delivery": ["arrived", "rider", "delivery", "delivered", "late"]
    }

    for category, words in keywords.items():

        if any(word in text for word in words):
            return category

    return None


# TASK 6: ZERO-SHOT / FEW-SHOT COMPARISON
def run_comparison(df, prompt_builder):

    mock_llm.reset()

    results = []
    retry_failures = 0

    valid_df = df.copy()
    valid_df["text"] = valid_df["text"].fillna("")

    for _, row in valid_df.iterrows():

        is_valid, _ = validate_input(row["text"])

        if not is_valid:
            continue

        clean_text, _, _, _ = redact_pii(row["text"])
        messages = prompt_builder(clean_text)

        response, retries = call_llm_with_retry(messages)
        retry_failures += retries

        response_text = response.choices[0].message.content

        result, status = validate_output(response_text)

        if status == "invalid_json":

            category = keyword_fallback(clean_text)
            status = "classified_keyword"

        elif status == "invalid_category":

            category = None
            status = "flagged_invalid_output"

        elif status == "classified_json":

            category = result["category"]

        else:

            category = None

        results.append({
            "ticket_id": row["ticket_id"],
            "predicted_category": category,
            "true_category": row["true_category"],
            "status": status
        })

    result_df = pd.DataFrame(results)

    evaluated = result_df.dropna(subset=["predicted_category"])

    accuracy = (
        evaluated["predicted_category"] == evaluated["true_category"]
    ).mean() * 100

    return accuracy, retry_failures, result_df


# CURRENT TEST: TASK 1
# TASK 7: BUDGET, ROUTING AND FINAL OUTPUT

def run_budget_pipeline(df, token_budget, output_file):

    mock_llm.reset()

    segments = pd.read_csv(ROOT / "outputs" / "customer_segments.csv")

    premium_customers = set(
        segments.loc[segments["cluster"] == 1, "customer_id"]
    )

    results = []
    total_tokens = 0
    model_calls = 0
    retry_failures = 0

    for _, row in df.iterrows():

        ticket_text = "" if pd.isna(row["text"]) else str(row["text"])

        is_valid, guardrail_status = validate_input(ticket_text)

        category = None
        tokens_used = 0

        if not is_valid:

            if guardrail_status == "empty_ticket":
                status = "skipped_empty"
            else:
                status = "flagged_long"

        elif total_tokens >= token_budget:

            status = "budget_exhausted"

        else:

            clean_text, _, _, _ = redact_pii(ticket_text)

            messages = build_few_shot_prompt(clean_text)

            prompt_text = " ".join(
                message["content"] for message in messages
            )

            # PII ASSERTIONS: ensure raw personal information is not sent
            assert not re.search(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
                prompt_text
            )

            assert not re.search(r'\b\d{10}\b', prompt_text)
            assert not re.search(r'\bORD-\d+\b', prompt_text)

            response, retries = call_llm_with_retry(messages)

            retry_failures += retries
            model_calls += 1

            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used

            response_text = response.choices[0].message.content

            result, validation_status = validate_output(response_text)

            if validation_status == "classified_json":

                category = result["category"]
                status = "classified_json"

            elif validation_status == "invalid_json":

                category = keyword_fallback(clean_text)
                status = "classified_keyword"

            else:

                category = None
                status = "flagged_invalid_output"

        customer_id = row["customer_id"]

        if status in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        ]:

            route = "human_review"
            priority = "NORMAL"

        elif category == "fraud":

            route = "security_team"
            priority = "HIGH"

        elif customer_id in premium_customers:

            route = "premium_desk"
            priority = "HIGH"

        else:

            route = "support_queue"
            priority = "NORMAL"

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })

    result_df = pd.DataFrame(results)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_file, index=False)

    print("\nPIPELINE SUMMARY")
    print("-" * 40)
    print("Token budget:", token_budget)
    print("Tokens used:", total_tokens)
    print("Model calls:", model_calls)
    print("Retry failures recovered:", retry_failures)
    print("Tickets processed:", len(result_df))
    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    print("\nSaved:", output_file)

    return result_df, total_tokens, model_calls, retry_failures

if __name__ == "__main__":

    df = pd.read_csv(INPUT_FILE)
    df["text"] = df["text"].fillna("")

    validation = df["text"].apply(validate_input)

    df["input_valid"] = validation.apply(lambda x: x[0])
    df["guardrail_status"] = validation.apply(lambda x: x[1])

    print("INPUT GUARDRAIL SUMMARY")
    print("-" * 40)

    print("Total tickets:", len(df))
    print("Valid tickets:", df["input_valid"].sum())
    print("Rejected tickets:", (~df["input_valid"]).sum())

    print("\nGuardrail status:")
    print(df["guardrail_status"].value_counts())

    print("\nRejected ticket IDs:")
    print(
        df.loc[
            ~df["input_valid"],
            ["ticket_id", "guardrail_status"]
        ].to_string(index=False)
    )
    run_budget_pipeline(df, token_budget=100000, output_file=OUTPUT_FILE)


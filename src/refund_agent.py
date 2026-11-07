from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_FILE = ROOT / "outputs" / "refund_agent_results.csv"

MAX_STEPS = 6


def get_order(order_id):
    orders = pd.read_csv(ORDERS_FILE)
    match = orders[orders["order_id"] == order_id]

    if match.empty:
        return {"error": "Order not found"}

    order = match.iloc[0]

    return {
        "order_id": order_id,
        "status": str(order["status"]),
        "order_amount": float(order["order_amount"]),
        "delivery_minutes": (
            None if pd.isna(order["delivery_minutes"])
            else float(order["delivery_minutes"])
        )
    }


def search_policy(query="refund policy"):
    policy_file = ROOT / "data" / "policies" / "refund_policy.txt"
    return {"policy": policy_file.read_text(encoding="utf-8")}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or pd.isna(delivery_minutes):
        refund = 0.0
    elif float(delivery_minutes) <= 45:
        refund = 0.0
    else:
        refund = round(float(order_amount), 2)

    return {"refund_inr": refund}


def issue_refund(order_id, refund_inr):
    return {
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "status": "issued"
    }


def escalate_to_human(order_id, refund_inr):
    return {
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "status": "escalated"
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order details",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search refund policy",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate eligible refund",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": ["number", "null"]}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an approved refund",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a large refund",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


def run_agent(message):
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are a safe refund assistant. "
                "Verify every order and follow the refund policy."
            )
        },
        {"role": "user", "content": message}
    ]

    steps = 0

    while steps < MAX_STEPS:
        response = mock_llm.chat(messages, tools=TOOLS)
        assistant_message = response.choices[0].message
        tool_calls = getattr(assistant_message, "tool_calls", None)

        if not tool_calls:
            return assistant_message.content, steps

        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": tool_calls
        })

        for tool_call in tool_calls:
            name = tool_call.name
            arguments = tool_call.arguments
            arguments = tool_call.arguments

            if name == "get_order":
                result = get_order(**arguments)

            elif name == "search_policy":
                result = search_policy(**arguments)

            elif name == "calculate_refund":
                result = calculate_refund(**arguments)

            elif name == "issue_refund":
                result = issue_refund(**arguments)

            elif name == "escalate_to_human":
                result = escalate_to_human(**arguments)

            else:
                result = {"error": "Unknown tool"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(result)
            })

        steps += 1

    return "FALLBACK: Maximum agent steps reached.", steps


def evaluate():
    scenarios = pd.read_csv(SCENARIOS_FILE)
    results = []

    for _, row in scenarios.iterrows():
        answer, steps = run_agent(row["message"])

        expected = row["expected_outcome"]

        if answer.startswith("REFUND:"):
            actual = "REFUND"
        elif answer.startswith("ESCALATE:"):
            actual = "ESCALATE"
        elif answer.startswith("DENIED:"):
            actual = "DENIED"
        elif answer.startswith("NOT_FOUND:"):
            actual = "NOT_FOUND"
        elif answer.startswith("FALLBACK:"):
            actual = "FALLBACK"
        else:
            actual = "NEEDS_INFO"

        results.append({
            "scenario_id": row["scenario_id"],
            "expected_outcome": expected,
            "actual_outcome": actual,
            "answer": answer,
            "steps": steps,
            "correct": actual == expected
        })

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nREFUND AGENT EVALUATION")
    print("----------------------")
    print("Total scenarios:", len(result_df))
    print("Correct outcomes:", result_df["correct"].sum())
    print("Accuracy:", round(result_df["correct"].mean() * 100, 2), "%")
    print("\nDETAILED RESULTS:")
    print(result_df.to_string(index=False))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate()
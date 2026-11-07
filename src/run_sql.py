
from pathlib import Path
import re
import sqlite3
import pandas as pd

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_queries():
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    # Split the SQL file into Q1 to Q7
    sections = re.split(r"(?=-- Q[1-7]:)", sql_text)

    queries = {}

    for section in sections:
        match = re.search(r"-- Q([1-7]):", section)
        if match:
            query_number = int(match.group(1))
            queries[query_number] = section.strip()

    if len(queries) != 7:
        raise ValueError(
            f"Expected 7 SQL queries, found {len(queries)}"
        )

    summaries = {}

    with sqlite3.connect(DB_PATH) as conn:
        for number in range(1, 8):
            print("\n" + "=" * 55)
            print(f"Q{number} RESULTS")
            print("=" * 55)

            df = pd.read_sql_query(
                queries[number], conn
            )

            output_file = RESULT_DIR / f"q{number}_results.csv"
            df.to_csv(output_file, index=False)

            print("Rows:", len(df))
            print(df.head(10).to_string(index=False))
            print("Saved:", output_file)

            summaries[number] = df

    # Reference checkpoint validation
    q1 = summaries[1]
    q2 = summaries[2]
    q3 = summaries[3]
    q4 = summaries[4]
    q5 = summaries[5]
    q6 = summaries[6]
    q7 = summaries[7]

    checks = {
        "Q1: 30 customers": len(q1) == 30,
        "Q2: 30 city-category rows": len(q2) == 30,
        "Q2: total revenue matches": abs(
            q2["revenue"].sum() - 2998668.80
        ) < 0.01,
        "Q3: 11 products": len(q3) == 11,
        "Q4: 25 customers": len(q4) == 25,
        "Q5: 550 customers": len(q5) == 550,
        "Q5: average spend matches": abs(
            q5["average_spend"].iloc[0] - 2552.06
        ) < 0.01,
        "Q7: 183 daily rows": len(q7) == 183,
        "Q7: final cumulative revenue matches": abs(
            q7["cumulative_revenue"].iloc[-1] - 2998668.80
        ) < 0.01,
    }

    print("\n" + "=" * 55)
    print("SQL REFERENCE VALIDATION")
    print("=" * 55)

    for description, passed in checks.items():
        print(f"{description}: {'PASS' if passed else 'FAIL'}")

    print("\nQ6: Top customer per city")
    top_customers = q6[q6["row_num"] == 1]
    print(
        top_customers[
            ["city", "customer_id", "total_spend",
             "spend_rank", "dense_spend_rank", "row_num"]
        ].to_string(index=False)
    )

    if all(checks.values()):
        print("\nALL SQL REFERENCE CHECKPOINTS PASSED")
    else:
        failed = [
            name for name, passed in checks.items() if not passed
        ]
        raise AssertionError(f"SQL validation failed: {failed}")


if __name__ == "__main__":
    run_queries()

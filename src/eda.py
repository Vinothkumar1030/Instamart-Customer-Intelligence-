
from pathlib import Path
import sqlite3
import pandas as pd

DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "instamart.db"
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    # Load cleaned data
    customers = pd.read_csv(CLEAN_DIR / "customers.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv")

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    # Keep completed orders for revenue analysis
    completed = orders[orders["status"] == "completed"].copy()

    # Join customer city and product category
    analysis = (
        completed
        .merge(customers[["customer_id", "city"]],
               on="customer_id", how="left")
        .merge(products[["product_id", "category"]],
               on="product_id", how="left")
    )

    # -------------------------------------------------
    # 1. Pandas pivot table cross-check with SQL Q2
    # -------------------------------------------------
    pivot = pd.pivot_table(
        analysis,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All"
    )

    print("\nPANDAS PIVOT TABLE – CITY AND CATEGORY REVENUE")
    print(pivot.round(2).to_string())

    pivot.to_csv(RESULT_DIR / "pandas_pivot_revenue.csv")

    sql_q2 = pd.read_csv(RESULT_DIR / "q2_results.csv")

    pivot_long = (
        analysis
        .groupby(["city", "category"], as_index=False)
        ["order_amount"].sum()
        .rename(columns={"order_amount": "pandas_revenue"})
    )

    comparison = sql_q2.merge(
        pivot_long,
        on=["city", "category"],
        how="outer"
    )

    comparison["difference"] = (
        comparison["revenue"] - comparison["pandas_revenue"]
    )

    print("\nSQL Q2 vs PANDAS CROSS-CHECK")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        RESULT_DIR / "sql_pandas_crosscheck.csv",
        index=False
    )

    # -------------------------------------------------
    # 2. City-wise grouped and named aggregation
    # -------------------------------------------------
    city_summary = (
        analysis
        .groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            average_order_amount=("order_amount", "mean"),
            completed_orders=("order_id", "count"),
            unique_customers=("customer_id", "nunique")
        )
        .reset_index()
    )

    print("\nCITY-WISE REVENUE SUMMARY")
    print(city_summary.round(2).to_string(index=False))

    city_summary.to_csv(
        RESULT_DIR / "city_revenue_summary.csv",
        index=False
    )

    # -------------------------------------------------
    # 3. Completed order descriptive statistics
    # -------------------------------------------------
    amount = completed["order_amount"].dropna()

    statistics = pd.DataFrame({
        "metric": [
            "count",
            "mean",
            "70th_percentile",
            "standard_deviation",
            "minimum",
            "maximum",
            "skewness"
        ],
        "value": [
            amount.count(),
            amount.mean(),
            amount.quantile(0.70),
            amount.std(),
            amount.min(),
            amount.max(),
            amount.skew()
        ]
    })

    print("\nCOMPLETED ORDER AMOUNT STATISTICS")
    print(statistics.round(2).to_string(index=False))

    statistics.to_csv(
        RESULT_DIR / "completed_order_statistics.csv",
        index=False
    )

    # -------------------------------------------------
    # 4. IQR outlier analysis
    # -------------------------------------------------
    q1 = amount.quantile(0.25)
    q3 = amount.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    below = (amount < lower_fence).sum()
    above = (amount > upper_fence).sum()

    print("\nIQR OUTLIER ANALYSIS")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower fence: {lower_fence:.2f}")
    print(f"Upper fence: {upper_fence:.2f}")
    print(f"Outliers below lower fence: {below}")
    print(f"Outliers above upper fence: {above}")

    # -------------------------------------------------
    # 5. Correlation analysis
    # -------------------------------------------------
    numeric_columns = [
        "order_amount",
        "quantity",
        "discount_pct"
    ]

    correlation = completed[numeric_columns].corr()

    print("\nCORRELATION MATRIX")
    print(correlation.round(3).to_string())

    correlation.to_csv(
        RESULT_DIR / "correlation_matrix.csv"
    )

    strongest = (
        correlation.abs()
        .where(~pd.DataFrame(
            [[i == j for j in range(len(correlation.columns))]
             for i in range(len(correlation.columns))],
            index=correlation.index,
            columns=correlation.columns
        ))
        .stack()
        .idxmax()
    )

    print(
        "\nStrongest absolute correlation:",
        strongest[0], "and", strongest[1],
        f"({correlation.loc[strongest[0], strongest[1]]:.3f})"
    )

    # -------------------------------------------------
    # 6. Validation
    # -------------------------------------------------
    checks = {
        "Pivot total matches completed revenue": abs(
            pivot.loc["All", "All"] - 2998668.80
        ) < 0.01,
        "SQL and Pandas revenue match": (
            comparison["difference"].abs().max() < 0.01
        ),
        "Completed order count": len(completed) == 11110,
    }

    print("\nEDA VALIDATION")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise AssertionError("EDA validation failed")

    print("\nALL EDA CHECKS PASSED")
    print("EDA results saved in:", RESULT_DIR)


if __name__ == "__main__":
    run_eda()

"""
SQL query analysis and Pandas equivalence verification.
Demonstrates: SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, and JOIN.
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "zepto_catalog.db"


def run_sql_queries():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at {DB_PATH}. Please run database.py first.")

    conn = sqlite3.connect(DB_PATH)

    queries = {
        "Query 1 (SELECT, WHERE, ORDER BY, LIMIT)": """
            SELECT title, price_gbp, price_inr, rating 
            FROM books 
            WHERE in_stock = 1 
            ORDER BY price_inr DESC 
            LIMIT 5;
        """,
        "Query 2 (DISTINCT)": """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating ASC;
        """,
        "Query 3 (BETWEEN)": """
            SELECT title, price_inr, rating 
            FROM books 
            WHERE price_inr BETWEEN 2000.0 AND 4000.0 
            ORDER BY price_inr ASC 
            LIMIT 5;
        """,
        "Query 4 (IN with subquery)": """
            SELECT title, rating, category_id 
            FROM books 
            WHERE category_id IN (
                SELECT category_id FROM categories WHERE category_name IN ('Travel', 'Mystery')
            )
            ORDER BY rating DESC 
            LIMIT 5;
        """,
        "Query 5 (JOIN across categories and books)": """
            SELECT b.title, c.category_name, b.price_inr, b.rating
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating >= 4
            ORDER BY b.price_inr DESC
            LIMIT 10;
        """
    }

    print("==================================================")
    print("EXECUTING REQUIRED SQL QUERIES")
    print("==================================================")
    for name, sql in queries.items():
        print(f"\n--- {name} ---")
        df_res = pd.read_sql_query(sql, conn)
        print(df_res.to_string(index=False))

    print("\n==================================================")
    print("VERIFYING PANDAS MERGE VS SQL JOIN EQUIVALENCE")
    print("==================================================")

    # 1. Via pd.read_sql
    sql_join_result = pd.read_sql_query(queries["Query 5 (JOIN across categories and books)"], conn)

    # 2. Via in-memory pd.merge
    books_df = pd.read_sql_query("SELECT * FROM books;", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories;", conn)

    merged_df = pd.merge(books_df, categories_df, on="category_id")
    filtered_df = merged_df[merged_df["rating"] >= 4]
    sorted_df = filtered_df.sort_values(by="price_inr", ascending=False).head(10)
    pandas_result = sorted_df[["title", "category_name", "price_inr", "rating"]].reset_index(drop=True)

    print("\n[Output 1: pd.read_sql]")
    print(sql_join_result.head())

    print("\n[Output 2: pd.merge in-memory]")
    print(pandas_result.head())

    # Assert programmatic equality
    pd.testing.assert_frame_equal(sql_join_result, pandas_result)
    print("\n[CHECK PASSED] Both outputs are structurally and numerically identical!")

    conn.close()


if __name__ == "__main__":
    run_sql_queries()
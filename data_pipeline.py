"""
Zepto Data Pipeline - Module 1 Complete Runner
Scrapes books.toscrape.com, cleans data, computes price_inr (105.50),
stores in normalized SQLite tables, and executes SQL + Pandas queries.
"""

import re
import sqlite3
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# 1. Setup Constants & Paths
DB_PATH = Path(__file__).resolve().parent / "zepto_catalog.db"
GBP_TO_INR_RATE = 105.50

TARGET_CATEGORIES = {
    "Travel": "http://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Mystery": "http://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "http://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "Sequential Art": "http://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
}

RATING_MAP = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


# 2. Scraper & Parser Functions
def parse_price(price_raw: str) -> float:
    try:
        clean_str = re.sub(r"[^\d.]", "", price_raw)
        return float(clean_str)
    except Exception:
        return np.nan


def parse_rating(class_list: list) -> int:
    for c in class_list:
        c_lower = c.lower()
        if c_lower in RATING_MAP:
            return RATING_MAP[c_lower]
    return np.nan


def scrape_category(category_name: str, category_url: str) -> list:
    current_url = category_url
    records = []

    while current_url:
        response = requests.get(current_url, timeout=15)
        if response.status_code != 200:
            print(f"Failed to fetch {current_url} (status: {response.status_code})")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        product_pods = soup.find_all("article", class_="product_pod")

        for pod in product_pods:
            title_tag = pod.h3.find("a")
            title = title_tag.get("title") or title_tag.text.strip()

            price_elem = pod.find("p", class_="price_color")
            raw_price = price_elem.text.strip() if price_elem else ""

            rating_elem = pod.find("p", class_="star-rating")
            rating_classes = rating_elem.get("class", []) if rating_elem else []
            rating = parse_rating(rating_classes)

            avail_elem = pod.find("p", class_="instock availability")
            avail_text = avail_elem.text.strip() if avail_elem else ""
            in_stock = 1 if "in stock" in avail_text.lower() else 0

            records.append({
                "title": title,
                "price_raw": raw_price,
                "rating": rating,
                "in_stock": in_stock,
                "category": category_name
            })

        next_button = soup.find("li", class_="next")
        if next_button and next_button.find("a"):
            next_href = next_button.find("a")["href"]
            current_url = current_url.rsplit("/", 1)[0] + "/" + next_href
        else:
            current_url = None

    return records


def run_scraping_and_cleaning() -> pd.DataFrame:
    raw_data = []
    print("\n[Step 1/3] Scraping books.toscrape.com...")
    for cat_name, url in TARGET_CATEGORIES.items():
        cat_books = scrape_category(cat_name, url)
        print(f"  - Scraped {len(cat_books)} books from '{cat_name}'")
        raw_data.extend(cat_books)

    df = pd.DataFrame(raw_data)
    print(f"Total raw records gathered: {len(df)}")

    # Clean & Convert Types
    df["price_gbp"] = df["price_raw"].apply(parse_price)

    if df["price_gbp"].isnull().any():
        med_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(med_price)

    if df["rating"].isnull().any():
        med_rating = int(df["rating"].median())
        df["rating"] = df["rating"].fillna(med_rating).astype(int)
    else:
        df["rating"] = df["rating"].astype(int)

    df["in_stock"] = df["in_stock"].astype(int)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)
    df = df.drop(columns=["price_raw"])
    return df


# 3. Database Schema and Population
def init_and_populate_db(df: pd.DataFrame):
    print("\n[Step 2/3] Initializing SQLite schema and loading records...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)

    # Populate categories
    unique_categories = df["category"].unique()
    for cat in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?);", (cat,))
    conn.commit()

    cursor.execute("SELECT category_name, category_id FROM categories;")
    cat_mapping = dict(cursor.fetchall())

    # Populate books
    books_data = [
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            cat_mapping[row["category"]]
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, books_data)

    conn.commit()
    conn.close()
    print(f"Stored {len(books_data)} books and {len(unique_categories)} categories into: {DB_PATH.name}")


# 4. SQL Queries and Pandas Equivalence Verification
def run_queries_and_assert():
    print("\n[Step 3/3] Executing 5 SQL queries and verifying Pandas equivalence...")
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

    for name, sql in queries.items():
        print(f"\n--- {name} ---")
        df_res = pd.read_sql_query(sql, conn)
        print(df_res.to_string(index=False))

    # Equivalence Verification
    print("\n--- Verifying pd.read_sql vs pd.merge Equivalence ---")
    sql_join_result = pd.read_sql_query(queries["Query 5 (JOIN across categories and books)"], conn)

    books_df = pd.read_sql_query("SELECT * FROM books;", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories;", conn)

    merged_df = pd.merge(books_df, categories_df, on="category_id")
    filtered_df = merged_df[merged_df["rating"] >= 4]
    sorted_df = filtered_df.sort_values(by="price_inr", ascending=False).head(10)
    pandas_result = sorted_df[["title", "category_name", "price_inr", "rating"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(sql_join_result, pandas_result)
    print("SUCCESS: pd.read_sql and pd.merge outputs are strictly identical!")

    conn.close()


if __name__ == "__main__":
    df_clean = run_scraping_and_cleaning()
    init_and_populate_db(df_clean)
    run_queries_and_assert()
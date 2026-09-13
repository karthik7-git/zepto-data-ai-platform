"""
Database module for creating normalized SQLite schema and inserting catalog data.
"""

"""
Database module for creating normalized SQLite schema and inserting catalog data.
"""

import sqlite3
from pathlib import Path
import pandas as pd
from scraper import run_scraping_pipeline

# Resolves to zepto_catalog.db inside the data_pipeline directory
DB_PATH = Path(__file__).resolve().parent / "zepto_catalog.db"


def init_db(db_file: Path = DB_PATH):
    """Initializes normalized SQLite schema with foreign key constraints enabled."""
    conn = sqlite3.connect(db_file)
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

    conn.commit()
    conn.close()
    print(f"Initialized normalized schema in: {db_file}")


def populate_db(df: pd.DataFrame, db_file: Path = DB_PATH):
    """Populates categories and books tables maintaining relational integrity."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    unique_categories = df["category"].unique()
    for cat in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?);", (cat,))
    conn.commit()

    cursor.execute("SELECT category_name, category_id FROM categories;")
    cat_mapping = dict(cursor.fetchall())

    books_data = []
    for _, row in df.iterrows():
        books_data.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            cat_mapping[row["category"]]
        ))

    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, books_data)

    conn.commit()
    print(f"Successfully inserted {len(books_data)} books and {len(unique_categories)} categories into {db_file.name}.")
    conn.close()


if __name__ == "__main__":
    df_clean = run_scraping_pipeline()
    init_db()
    populate_db(df_clean)
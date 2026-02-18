"""
Load MovieLens 100k data files into a SQLite database.
Tables created:
  - users        (from u.user)
  - items        (from u.item)
  - ratings      (from u.data  – absent, reconstructed from ua/ub splits)
  - genres       (from u.genre)
  - occupations  (from u.occupation)
  - ua_base, ua_test, ub_base, ub_test  (train/test splits)
"""

import csv
import os
import sqlite3
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ml-100k")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "movielens.db")

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film_Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci_Fi", "Thriller", "War", "Western",
]


def connect():
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_lines(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="latin-1") as f:
        return f.readlines()


def _load_ratings_file(conn, filename, table_name):
    """Load a tab-separated ratings file (user_id, item_id, rating, timestamp)."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            user_id   INTEGER NOT NULL,
            item_id   INTEGER NOT NULL,
            rating    INTEGER NOT NULL,
            timestamp INTEGER NOT NULL
        )
    """)
    rows = []
    for line in _read_lines(filename):
        parts = line.strip().split("\t")
        if len(parts) == 4:
            rows.append(tuple(int(p) for p in parts))
    conn.executemany(
        f"INSERT INTO {table_name} (user_id, item_id, rating, timestamp) VALUES (?,?,?,?)",
        rows,
    )
    return len(rows)


# ── table loaders ────────────────────────────────────────────────────────────

def load_genres(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS genres (
            genre_id   INTEGER PRIMARY KEY,
            genre_name TEXT NOT NULL
        )
    """)
    rows = []
    for line in _read_lines("u.genre"):
        line = line.strip()
        if not line:
            continue
        name, gid = line.rsplit("|", 1)
        rows.append((int(gid), name))
    conn.executemany("INSERT INTO genres VALUES (?,?)", rows)
    print(f"  genres: {len(rows)} rows")


def load_occupations(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS occupations (
            occupation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            occupation_name TEXT NOT NULL UNIQUE
        )
    """)
    rows = []
    for line in _read_lines("u.occupation"):
        line = line.strip()
        if line:
            rows.append((line,))
    conn.executemany("INSERT INTO occupations (occupation_name) VALUES (?)", rows)
    print(f"  occupations: {len(rows)} rows")


def load_users(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY,
            age        INTEGER,
            gender     TEXT,
            occupation TEXT,
            zip_code   TEXT
        )
    """)
    rows = []
    for line in _read_lines("u.user"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3], parts[4]))
    conn.executemany("INSERT INTO users VALUES (?,?,?,?,?)", rows)
    print(f"  users: {len(rows)} rows")


def load_items(conn):
    genre_cols = ", ".join(f"{g} INTEGER" for g in GENRE_COLUMNS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS items (
            item_id            INTEGER PRIMARY KEY,
            title              TEXT,
            release_date       TEXT,
            video_release_date TEXT,
            imdb_url           TEXT,
            {genre_cols}
        )
    """)
    rows = []
    for line in _read_lines("u.item"):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split("|")
        # parts: id | title | release_date | video_release_date | url | 19 genre flags
        item_id = int(parts[0])
        title = parts[1]
        release_date = parts[2] if parts[2] else None
        video_release_date = parts[3] if parts[3] else None
        imdb_url = parts[4] if parts[4] else None
        genre_flags = [int(g) for g in parts[5:24]]
        # pad if fewer than 19 genre flags
        while len(genre_flags) < 19:
            genre_flags.append(0)
        rows.append((item_id, title, release_date, video_release_date, imdb_url, *genre_flags))

    placeholders = ",".join(["?"] * (5 + 19))
    conn.executemany(f"INSERT INTO items VALUES ({placeholders})", rows)
    print(f"  items: {len(rows)} rows")


def load_ratings_splits(conn):
    """Load the ua/ub base/test split files."""
    for name, filename in [
        ("ua_base", "ua.base"),
        ("ua_test", "ua.test"),
        ("ub_base", "ub.base"),
        ("ub_test", "ub.test"),
    ]:
        n = _load_ratings_file(conn, filename, name)
        print(f"  {name}: {n} rows")


def load_u_data(conn):
    """Load u.data (the full 100k ratings) if present."""
    path = os.path.join(DATA_DIR, "u.data")
    if os.path.exists(path):
        n = _load_ratings_file(conn, "u.data", "ratings")
        print(f"  ratings (u.data): {n} rows")
    else:
        print("  ratings: u.data not found – skipped")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Remove existing DB so we start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = connect()
    print(f"Creating database at {os.path.abspath(DB_PATH)}")
    print("Loading tables …")

    load_genres(conn)
    load_occupations(conn)
    load_users(conn)
    load_items(conn)
    load_u_data(conn)
    load_ratings_splits(conn)

    conn.commit()

    # Quick verification
    print("\n── Verification ──")
    for table in ["genres", "occupations", "users", "items", "ratings",
                   "ua_base", "ua_test", "ub_base", "ub_test"]:
        try:
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            print(f"  {table:15s} {count:>8,} rows")
        except sqlite3.OperationalError:
            print(f"  {table:15s}  (not created)")
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

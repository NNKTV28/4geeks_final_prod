"""
01 — Carga de Datos y Limpieza
===============================
Carga tablas de MovieLens 100K desde SQLite, unifica datasets,
analiza valores nulos, duplicados y estadísticas descriptivas.
"""

import os
import sqlite3
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "..", "data", "movielens.db")


# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Carga ratings, users y movies desde SQLite y devuelve el df combinado."""
    print("══════════════════════════════════════════════════════════════")
    print("  PROYECTO MOVIELENS 100K — CARGA DE DATOS")
    print("══════════════════════════════════════════════════════════════")
    print(f"\n  Conectando a: {os.path.abspath(DB_PATH)}")

    conn = sqlite3.connect(DB_PATH)

    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    ratings.columns = ["userId", "movieId", "rating", "timestamp"]
    print(f"  Ratings: {len(ratings):,} filas")

    users = pd.read_sql("SELECT * FROM users", conn)
    users.columns = ["userId", "age", "gender", "occupation", "zipcode"]
    print(f"  Users:   {len(users):,} filas")

    movies = pd.read_sql("SELECT * FROM items", conn)
    movies.rename(columns={"item_id": "movieId"}, inplace=True)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    print(f"  Movies:  {len(movies):,} filas")

    conn.close()

    # Combinar datasets
    df = ratings.merge(users, on="userId", how="left")
    df = df.merge(movies, on="movieId", how="left")
    print(f"\n  Dataset completo: {len(df):,} filas, {len(df.columns)} columnas")

    return df, movies


# ═══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE CALIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def analisis_nulos(df):
    """Revisa y muestra valores nulos en el dataset."""
    print("\n── VALORES NULOS ──")
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df)) * 100
    nulos_df = pd.DataFrame({
        "Columna": nulos.index,
        "Nulos": nulos.values,
        "Porcentaje": nulos_pct.values,
    })
    nulos_df = nulos_df[nulos_df["Nulos"] > 0].sort_values("Nulos", ascending=False)

    if len(nulos_df) > 0:
        print(nulos_df.to_string(index=False))
    else:
        print("  No hay valores nulos")


def analisis_duplicados(df):
    """Revisa y muestra filas duplicadas."""
    print("\n── DUPLICADOS ──")
    duplicados = df.duplicated().sum()
    print(f"  Filas duplicadas: {duplicados} ({duplicados / len(df) * 100:.2f}%)")

    if duplicados > 0:
        print("\n  Ejemplo de duplicados:")
        print(df[df.duplicated(keep=False)].head())


def estadisticas_descriptivas(df):
    """Muestra estadísticas descriptivas clave."""
    print("\n── ESTADÍSTICAS DESCRIPTIVAS ──")
    print(
        f"  RATINGS  → Media: {df['rating'].mean():.2f} | "
        f"Mediana: {df['rating'].median():.1f} | "
        f"Moda: {df['rating'].mode()[0]:.0f} | "
        f"σ: {df['rating'].std():.2f} | "
        f"Rango: {df['rating'].min():.0f}–{df['rating'].max():.0f}"
    )
    print(
        f"  EDAD     → Media: {df['age'].mean():.1f} | "
        f"Rango: {df['age'].min():.0f}–{df['age'].max():.0f}"
    )
    print(
        f"  AÑO PELIS→ {df['year'].min():.0f}–{df['year'].max():.0f} | "
        f"Promedio: {df['year'].mean():.0f}"
    )


def distribuciones(df):
    """Muestra distribuciones de variables categóricas."""
    print("\n── DISTRIBUCIONES ──")
    print("\nRatings:")
    print(df["rating"].value_counts().sort_index().to_string())
    print("\nGénero:")
    print(df["gender"].value_counts().to_string())
    print("\nTop 10 Ocupaciones:")
    print(df["occupation"].value_counts().head(10).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    """Ejecuta la carga y limpieza completa."""
    df, movies = load_data()
    analisis_nulos(df)
    analisis_duplicados(df)
    estadisticas_descriptivas(df)
    distribuciones(df)
    return df, movies


if __name__ == "__main__":
    run()

"""
04 — Evaluación del Sistema de Popularidad
============================================
Evalúa un sistema de recomendación basado en popularidad usando
5 métricas de ranking: Precision@K, Recall@K, MAP@K, NDCG@K y Coverage.

Guarda resultados en  src/eda_completo/metricas_popularidad.csv
"""

import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "..", "data", "movielens.db")


# ═══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def precision_at_k(recommended, relevant, k=10):
    """Precision@K: de las K recomendadas, cuántas son relevantes."""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = len([r for r in recommended_k if r in relevant_set])
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k=10):
    """Recall@K: de todos los items relevantes, cuántos logré recomendar."""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = len([r for r in recommended_k if r in relevant_set])
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def average_precision(recommended, relevant):
    """Average Precision: precisión considerando el orden."""
    relevant_set = set(relevant)
    precisions = []
    hits = 0
    for i, item in enumerate(recommended):
        if item in relevant_set:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.mean(precisions) if precisions else 0.0


def map_at_k(all_recommended, all_relevant, k=10):
    """MAP@K: Mean Average Precision promediado por usuarios."""
    aps = []
    for recommended, relevant in zip(all_recommended, all_relevant):
        recommended_k = recommended[:k]
        ap = average_precision(recommended_k, relevant)
        aps.append(ap)
    return np.mean(aps) if aps else 0.0


def ndcg_at_k(recommended, relevant, k=10):
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)

    ideal_recommended = relevant[:k]
    idcg = 0.0
    for i in range(len(ideal_recommended)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommended, total_items):
    """Coverage: porcentaje del catálogo cubierto por recomendaciones."""
    unique_recommended = set()
    for recommended in all_recommended:
        unique_recommended.update(recommended)
    return len(unique_recommended) / total_items if total_items > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  SISTEMA DE POPULARIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    """Ejecuta la evaluación del sistema de popularidad."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  EVALUACIÓN — Sistema de Popularidad")
    print("══════════════════════════════════════════════════════════════")

    conn = sqlite3.connect(DB_PATH)
    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    ratings.columns = ["userId", "movieId", "rating", "timestamp"]
    conn.close()

    # Train / test split
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_data):,} ratings")
    print(f"  Test:  {len(test_data):,} ratings")

    # Top 100 películas populares por train
    movie_pop = train_data.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
    ).reset_index()
    movie_pop = movie_pop[movie_pop["num_ratings"] >= 5]
    movie_pop = movie_pop.sort_values("avg_rating", ascending=False)
    top_popular = movie_pop["movieId"].head(100).tolist()

    # Evaluar sobre 100 usuarios de test
    test_users = test_data["userId"].unique()[:100]
    all_recommended_pop = []
    all_relevant_pop = []

    for user in test_users:
        relevant = test_data[
            (test_data["userId"] == user) & (test_data["rating"] >= 4)
        ]["movieId"].tolist()
        if len(relevant) > 0:
            all_recommended_pop.append(top_popular)
            all_relevant_pop.append(relevant)

    print(f"  Usuarios evaluados: {len(all_recommended_pop)}")

    # Calcular métricas
    precision_scores = [
        precision_at_k(rec, rel, 10)
        for rec, rel in zip(all_recommended_pop, all_relevant_pop)
    ]
    precision_pop = np.mean(precision_scores)

    recall_scores = [
        recall_at_k(rec, rel, 10)
        for rec, rel in zip(all_recommended_pop, all_relevant_pop)
    ]
    recall_pop = np.mean(recall_scores)

    map_pop = map_at_k(all_recommended_pop, all_relevant_pop, 10)

    ndcg_scores = [
        ndcg_at_k(rec, rel, 10)
        for rec, rel in zip(all_recommended_pop, all_relevant_pop)
    ]
    ndcg_pop = np.mean(ndcg_scores)

    total_movies = ratings["movieId"].nunique()
    coverage_pop = coverage(all_recommended_pop, total_movies)

    # Mostrar resultados
    print("\n  RESULTADOS — SISTEMA DE POPULARIDAD")
    print("=" * 55)
    print(f"  Precision@10:  {precision_pop:.4f}")
    print(f"  Recall@10:     {recall_pop:.4f}")
    print(f"  MAP@10:        {map_pop:.4f}")
    print(f"  NDCG@10:       {ndcg_pop:.4f}")
    print(f"  Coverage:      {coverage_pop:.4f}")
    print("=" * 55)

    # Interpretación
    print(f"""
  Precision@10 = {precision_pop:.2%}
    De cada 10 películas recomendadas, {precision_pop * 10:.1f} le gustan al usuario

  Recall@10 = {recall_pop:.2%}
    De todas las películas que le gustan, encontramos {recall_pop * 100:.1f}%

  MAP@10 = {map_pop:.2%}
    Calidad del ordenamiento de las recomendaciones

  NDCG@10 = {ndcg_pop:.2%}
    Calidad del ranking (considerando el orden ideal)

  Coverage = {coverage_pop:.2%}
    Recomendamos {coverage_pop * 100:.1f}% del catálogo total
""")

    # Guardar resultados
    resultados = {
        "sistema": "Popularidad",
        "precision_10": precision_pop,
        "recall_10": recall_pop,
        "map_10": map_pop,
        "ndcg_10": ndcg_pop,
        "coverage": coverage_pop,
    }

    out_csv = os.path.join(SCRIPT_DIR, "metricas_popularidad.csv")
    pd.DataFrame([resultados]).to_csv(out_csv, index=False)

    out_pkl = os.path.join(SCRIPT_DIR, "metricas_popularidad.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(resultados, f)

    print(f"  Guardado: {out_csv}")
    print(f"  Guardado: {out_pkl}")

    print("\n  Evaluación de popularidad completada.")


if __name__ == "__main__":
    run()

"""
PROYECTO MOVIELENS 100K — EDA + Evaluación de 3 Sistemas de Recomendación
=========================================================================
Ejecutar:  python src/evaluate_models.py
Genera todos los gráficos (.png) en src/ y muestra resultados en consola.
"""

import os
import sys
import sqlite3
import warnings

# Asegurar que la consola de Windows soporte UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "data", "movielens.db")
OUT_DIR = os.path.join(PROJECT_DIR, "images")  # imágenes en carpeta dedicada
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTE 1 — CARGA DE DATOS Y EDA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Carga ratings, users y movies desde SQLite y devuelve el df combinado."""
    print("══════════════════════════════════════════════════════════════")
    print("  PROYECTO MOVIELENS 100K")
    print("══════════════════════════════════════════════════════════════")
    print(f"\n  Conectando a: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    ratings.columns = ["userId", "movieId", "rating", "timestamp"]
    print(f"  Ratings: {len(ratings):,} filas")

    users = pd.read_sql("SELECT * FROM users", conn)
    users.columns = ["userId", "age", "gender", "occupation", "zipcode"]
    print(f"  Users: {len(users):,} filas")

    movies = pd.read_sql("SELECT * FROM items", conn)
    movies.rename(columns={"item_id": "movieId"}, inplace=True)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    print(f"  Movies: {len(movies):,} filas")

    conn.close()

    # Combinar datasets
    df = ratings.merge(users, on="userId", how="left")
    df = df.merge(movies, on="movieId", how="left")
    print(f"\n  Dataset completo: {len(df):,} filas, {len(df.columns)} columnas")
    return df, movies


def eda_basico(df):
    """Valores nulos, duplicados y estadísticas descriptivas."""
    # ── Valores nulos ──
    print("\n── VALORES NULOS ──")
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df)) * 100
    nulos_df = pd.DataFrame({"Columna": nulos.index, "Nulos": nulos.values, "Porcentaje": nulos_pct.values})
    nulos_df = nulos_df[nulos_df["Nulos"] > 0].sort_values("Nulos", ascending=False)
    if len(nulos_df) > 0:
        print(nulos_df.to_string(index=False))
    else:
        print("  ✅ No hay valores nulos")

    # ── Duplicados ──
    print("\n── DUPLICADOS ──")
    duplicados = df.duplicated().sum()
    print(f"  Filas duplicadas: {duplicados} ({duplicados / len(df) * 100:.2f}%)")

    # ── Estadísticas descriptivas ──
    print("\n── ESTADÍSTICAS DESCRIPTIVAS ──")
    print(f"  RATINGS  → Media: {df['rating'].mean():.2f} | Mediana: {df['rating'].median():.1f} "
          f"| Moda: {df['rating'].mode()[0]:.0f} | σ: {df['rating'].std():.2f} "
          f"| Rango: {df['rating'].min():.0f}–{df['rating'].max():.0f}")
    print(f"  EDAD     → Media: {df['age'].mean():.1f} | Rango: {df['age'].min():.0f}–{df['age'].max():.0f}")
    print(f"  AÑO PELIS→ {df['year'].min():.0f}–{df['year'].max():.0f} | Promedio: {df['year'].mean():.0f}")

    # ── Distribuciones ──
    print("\n── DISTRIBUCIONES ──")
    print("\nRatings:")
    print(df["rating"].value_counts().sort_index().to_string())
    print("\nGénero:")
    print(df["gender"].value_counts().to_string())
    print("\nTop 10 Ocupaciones:")
    print(df["occupation"].value_counts().head(10).to_string())


def eda_visualizaciones(df):
    """Genera el grid de 8 visualizaciones EDA."""
    print("\n  Generando visualizaciones EDA...")
    fig = plt.figure(figsize=(16, 12))

    # 1. Distribución de Ratings
    ax1 = plt.subplot(3, 3, 1)
    df["rating"].value_counts().sort_index().plot(kind="bar", color="red", ax=ax1)
    ax1.set_title("Distribución de Ratings", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rating"); ax1.set_ylabel("Frecuencia")

    # 2. Distribución de Edad
    ax2 = plt.subplot(3, 3, 2)
    df["age"].hist(bins=30, color="blue", edgecolor="black", ax=ax2)
    ax2.set_title("Distribución de Edad", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Edad"); ax2.set_ylabel("Frecuencia")

    # 3. Rating por Género
    ax3 = plt.subplot(3, 3, 3)
    df.groupby("gender")["rating"].mean().plot(kind="bar", color="red", ax=ax3)
    ax3.set_title("Rating Promedio por Género", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Género"); ax3.set_ylabel("Rating Promedio")
    ax3.set_xticklabels(["F", "M"], rotation=0)

    # 4. Ratings por Año
    ax4 = plt.subplot(3, 3, 4)
    df.groupby("year").size().plot(color="blue", linewidth=2, ax=ax4)
    ax4.set_title("Nº de Ratings por Año de Película", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Año"); ax4.set_ylabel("Nº de Ratings")

    # 5. Top 10 Ocupaciones
    ax5 = plt.subplot(3, 3, 5)
    df["occupation"].value_counts().head(10).plot(kind="barh", color="red", ax=ax5)
    ax5.set_title("Top 10 Ocupaciones", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Frecuencia")

    # 6. Rating por Edad
    ax6 = plt.subplot(3, 3, 6)
    age_bins = [0, 18, 25, 35, 50, 100]
    age_labels = ["<18", "18-24", "25-34", "35-49", "50+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    df.groupby("age_group")["rating"].mean().plot(kind="bar", color="blue", ax=ax6)
    ax6.set_title("Rating Promedio por Grupo de Edad", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Grupo de Edad"); ax6.set_ylabel("Rating Promedio")
    ax6.set_xticklabels(age_labels, rotation=45)

    # 7. Películas por Década
    df["decade"] = (df["year"] // 10) * 10
    ax8 = plt.subplot(3, 3, 8)
    df.groupby("decade").size().plot(kind="bar", color="red", ax=ax8)
    ax8.set_title("Películas por Década", fontsize=12, fontweight="bold")
    ax8.set_xlabel("Década"); ax8.set_ylabel("Número de Películas")

    # 8. Top Géneros
    ax9 = plt.subplot(3, 3, 9)
    genre_cols_short = ["Action", "Comedy", "Drama", "Sci_Fi", "Thriller", "Horror", "Romance"]
    genre_counts = df[genre_cols_short].sum().sort_values(ascending=True)
    genre_counts.plot(kind="barh", color="blue", ax=ax9)
    ax9.set_title("Top Géneros", fontsize=12, fontweight="bold")
    ax9.set_xlabel("Número de Películas")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eda_visualizations.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  → Guardado: {path}")
    return df


def eda_correlaciones(df):
    """Heatmap de correlaciones."""
    numeric_cols = ["rating", "age", "year"]
    correlation_matrix = df[numeric_cols].corr()
    print("\n── MATRIZ DE CORRELACIÓN ──")
    print(correlation_matrix.to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn", center=0,
                square=True, linewidths=1, cbar=True)
    plt.title("Matriz de Correlación", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "correlation_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  → Guardado: {path}")


def eda_temporal(df):
    """Análisis temporal por década + top películas."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  EDA NIKITA — 1. Análisis Temporal")
    print("══════════════════════════════════════════════════════════════")

    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    movies_per_decade = df.drop_duplicates("movieId").groupby("decade").size().reset_index(name="num_movies")
    ratings_per_decade = df.groupby("decade").size().reset_index(name="num_ratings")
    avg_rating_decade = df.groupby("decade")["rating"].mean().reset_index(name="avg_rating")
    decade_summary = movies_per_decade.merge(ratings_per_decade, on="decade").merge(avg_rating_decade, on="decade")
    decade_summary["avg_rating"] = decade_summary["avg_rating"].round(2)
    print(decade_summary.to_string(index=False))

    # Visualización por década
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(decade_summary["decade"].astype(str), decade_summary["num_movies"], color="#2196F3", edgecolor="white")
    axes[0].set_title("Número de Películas por Década", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Década"); axes[0].set_ylabel("Cantidad de Películas")
    for i, v in enumerate(decade_summary["num_movies"]):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=9)

    axes[1].bar(decade_summary["decade"].astype(str), decade_summary["num_ratings"], color="#FF9800", edgecolor="white")
    axes[1].set_title("Número de Ratings por Década", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Década"); axes[1].set_ylabel("Cantidad de Ratings")
    for i, v in enumerate(decade_summary["num_ratings"]):
        axes[1].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold", fontsize=9)

    axes[2].plot(decade_summary["decade"].astype(str), decade_summary["avg_rating"],
                 marker="o", linewidth=2.5, color="#4CAF50", markersize=8)
    axes[2].set_title("Rating Promedio por Década", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Década"); axes[2].set_ylabel("Rating Promedio")
    axes[2].set_ylim(2.5, 4.5)
    axes[2].axhline(y=df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                     label=f'Media global: {df["rating"].mean():.2f}')
    axes[2].legend()
    plt.tight_layout()
    plt.show()

    # Top 3 películas por década
    top_by_decade = (
        df.groupby(["decade", "movieId", "title"])
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
        .query("num_ratings >= 20")
        .sort_values(["decade", "avg_rating"], ascending=[True, False])
        .groupby("decade")
        .head(3)
    )
    print("\n  TOP 3 PELÍCULAS POR DÉCADA (mín. 20 ratings):\n")
    for decade, group in top_by_decade.groupby("decade"):
        print(f"  Década {decade}s:")
        for _, row in group.iterrows():
            print(f"    {row['title']} → ★ {row['avg_rating']:.2f} ({row['num_ratings']} ratings)")
        print()


def eda_generos(df):
    """Análisis por género de película."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  EDA NIKITA — 2. Análisis por Género de Película")
    print("══════════════════════════════════════════════════════════════")

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Childrens",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film_Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci_Fi", "Thriller", "War", "Western",
    ]

    genre_stats = []
    for g in genre_cols:
        if g in df.columns:
            subset = df[df[g] == 1]
            genre_stats.append({
                "genre": g,
                "num_movies": subset["movieId"].nunique(),
                "num_ratings": len(subset),
                "avg_rating": round(subset["rating"].mean(), 2),
            })
    genre_df = pd.DataFrame(genre_stats).sort_values("num_ratings", ascending=False)
    print(genre_df.to_string(index=False))

    # Visualización popularidad y rating por género
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    genre_sorted = genre_df.sort_values("num_ratings")
    colors_pop = plt.cm.Blues(np.linspace(0.3, 0.9, len(genre_sorted)))
    axes[0].barh(genre_sorted["genre"], genre_sorted["num_ratings"], color=colors_pop, edgecolor="white")
    axes[0].set_title("Popularidad por Género (Nº de Ratings)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Cantidad de Ratings")
    for i, v in enumerate(genre_sorted["num_ratings"]):
        axes[0].text(v + 200, i, f"{v:,}", va="center", fontsize=8)

    genre_sorted_rating = genre_df.sort_values("avg_rating")
    colors_rat = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(genre_sorted_rating)))
    axes[1].barh(genre_sorted_rating["genre"], genre_sorted_rating["avg_rating"], color=colors_rat, edgecolor="white")
    axes[1].set_title("Rating Promedio por Género", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Rating Promedio")
    axes[1].set_xlim(2.5, 4.5)
    axes[1].axvline(x=df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                     label=f'Media global: {df["rating"].mean():.2f}')
    axes[1].legend()
    for i, v in enumerate(genre_sorted_rating["avg_rating"]):
        axes[1].text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()

    # Co-ocurrencia de géneros
    movies_unique = df.drop_duplicates("movieId")
    genre_matrix = movies_unique[genre_cols].astype(int)
    co_occurrence = genre_matrix.T.dot(genre_matrix)
    co_vals = co_occurrence.values.copy()
    np.fill_diagonal(co_vals, 0)
    co_occurrence = pd.DataFrame(co_vals, index=co_occurrence.index, columns=co_occurrence.columns)

    plt.figure(figsize=(12, 9))
    sns.heatmap(co_occurrence, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=genre_cols, yticklabels=genre_cols,
                linewidths=0.5, linecolor="white")
    plt.title("Co-ocurrencia de Géneros en Películas", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Evolución de géneros por década
    genre_decade = []
    for g in genre_cols:
        if g in df.columns:
            temp = df[df[g] == 1].groupby("decade").agg(
                avg_rating=("rating", "mean"), count=("rating", "count")
            ).reset_index()
            temp["genre"] = g
            genre_decade.append(temp)
    genre_decade_df = pd.concat(genre_decade, ignore_index=True)

    top5_genres = genre_df.head(5)["genre"].tolist()
    gd_top5 = genre_decade_df[genre_decade_df["genre"].isin(top5_genres)]

    plt.figure(figsize=(12, 5))
    for genre in top5_genres:
        subset = gd_top5[gd_top5["genre"] == genre]
        plt.plot(subset["decade"].astype(str), subset["avg_rating"], marker="o", linewidth=2, label=genre)
    plt.title("Evolución del Rating Promedio por Género (Top 5)", fontsize=13, fontweight="bold")
    plt.xlabel("Década"); plt.ylabel("Rating Promedio")
    plt.legend(title="Género", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return genre_df


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTE 2 — EVALUACIÓN DE 3 SISTEMAS DE RECOMENDACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def load_train_test():
    """Carga los splits ua_base / ua_test para evaluación."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  EVALUACIÓN DE MODELOS — Comparación de 3 Sistemas")
    print("══════════════════════════════════════════════════════════════")

    conn = sqlite3.connect(DB_PATH)
    train = pd.read_sql("SELECT * FROM ua_base", conn)
    test = pd.read_sql("SELECT * FROM ua_test", conn)
    train.columns = ["userId", "movieId", "rating", "timestamp"]
    test.columns = ["userId", "movieId", "rating", "timestamp"]
    movies = pd.read_sql("SELECT item_id as movieId, title FROM items", conn)
    conn.close()

    n_users = max(train["userId"].max(), test["userId"].max())
    n_items = max(train["movieId"].max(), test["movieId"].max())

    print(f"  Train: {len(train):,} ratings")
    print(f"  Test:  {len(test):,} ratings")
    print(f"  Usuarios: {n_users} | Películas: {n_items}")
    print(f"  Sparsity: {1 - len(train) / (n_users * n_items):.2%}")

    return train, test, movies, n_users, n_items


def build_matrices(train, n_users, n_items):
    """Construye la matriz usuario-ítem y la versión centrada."""
    user_item_matrix = np.zeros((n_users, n_items))
    for _, row in train.iterrows():
        user_item_matrix[int(row["userId"]) - 1, int(row["movieId"]) - 1] = row["rating"]

    user_means = np.zeros(n_users)
    for u in range(n_users):
        rated = user_item_matrix[u, :] > 0
        if rated.sum() > 0:
            user_means[u] = user_item_matrix[u, rated].mean()

    user_item_centered = user_item_matrix.copy()
    for u in range(n_users):
        rated = user_item_matrix[u, :] > 0
        user_item_centered[u, rated] -= user_means[u]

    print(f"\n  Matriz usuario-ítem: {user_item_matrix.shape}")
    print(f"  Entradas no-cero: {(user_item_matrix > 0).sum():,}")

    return user_item_matrix, user_item_centered, user_means


def evaluate_user_cf(test, user_item_matrix, user_item_centered, user_means, n_users, n_items, k=30):
    """Sistema 1: User-Based Collaborative Filtering."""
    print("\n── Sistema 1: User-Based CF ──")
    user_similarity = cosine_similarity(user_item_centered)
    np.fill_diagonal(user_similarity, 0)

    def predict(user_id, item_id):
        u, i = user_id - 1, item_id - 1
        rated_mask = user_item_matrix[:, i] > 0
        if not rated_mask.any():
            return user_means[u] if user_means[u] > 0 else 3.0
        sims = user_similarity[u, :] * rated_mask
        top_k_idx = np.argsort(sims)[-k:]
        top_k_sims = sims[top_k_idx]
        top_k_ratings = user_item_centered[top_k_idx, i]
        denom = np.abs(top_k_sims).sum()
        if denom == 0:
            return user_means[u] if user_means[u] > 0 else 3.0
        return np.clip(user_means[u] + np.dot(top_k_sims, top_k_ratings) / denom, 1, 5)

    print("  Evaluando... (puede tardar ~1 min)")
    preds, actuals = [], []
    for _, row in test.iterrows():
        uid, mid, actual = int(row["userId"]), int(row["movieId"]), row["rating"]
        if uid <= n_users and mid <= n_items:
            preds.append(predict(uid, mid))
            actuals.append(actual)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    print(f"  User-Based CF → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return preds, actuals, rmse, mae


def evaluate_item_cf(test, user_item_matrix, user_item_centered, user_means, n_users, n_items, k=30):
    """Sistema 2: Item-Based Collaborative Filtering."""
    print("\n── Sistema 2: Item-Based CF ──")
    item_similarity = cosine_similarity(user_item_centered.T)
    np.fill_diagonal(item_similarity, 0)

    def predict(user_id, item_id):
        u, i = user_id - 1, item_id - 1
        rated_mask = user_item_matrix[u, :] > 0
        if not rated_mask.any():
            return user_means[u] if user_means[u] > 0 else 3.0
        sims = item_similarity[i, :] * rated_mask
        top_k_idx = np.argsort(sims)[-k:]
        top_k_sims = sims[top_k_idx]
        top_k_ratings = user_item_matrix[u, top_k_idx]
        denom = np.abs(top_k_sims).sum()
        if denom == 0:
            return user_means[u] if user_means[u] > 0 else 3.0
        return np.clip(np.dot(top_k_sims, top_k_ratings) / denom, 1, 5)

    print("  Evaluando... (puede tardar ~1 min)")
    preds, actuals = [], []
    for _, row in test.iterrows():
        uid, mid, actual = int(row["userId"]), int(row["movieId"]), row["rating"]
        if uid <= n_users and mid <= n_items:
            preds.append(predict(uid, mid))
            actuals.append(actual)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    print(f"  Item-Based CF → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return preds, actuals, rmse, mae


def evaluate_svd(test, user_item_centered, user_means, n_users, n_items, n_factors=50):
    """Sistema 3: SVD (Matrix Factorization)."""
    print("\n── Sistema 3: SVD ──")
    U, sigma, Vt = svds(user_item_centered.astype(float), k=n_factors)
    sigma_diag = np.diag(sigma)
    svd_predictions = user_means.reshape(-1, 1) + U.dot(sigma_diag).dot(Vt)
    svd_predictions = np.clip(svd_predictions, 1, 5)

    print("  Evaluando...")
    preds, actuals = [], []
    for _, row in test.iterrows():
        uid, mid, actual = int(row["userId"]), int(row["movieId"]), row["rating"]
        if uid <= n_users and mid <= n_items:
            preds.append(svd_predictions[uid - 1, mid - 1])
            actuals.append(actual)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    print(f"  SVD ({n_factors} factores) → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return preds, actuals, rmse, mae


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTE 3 — TABLA COMPARATIVA Y GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

def precision_recall_at_k(actuals, preds, user_ids, k=10, threshold=4.0):
    """Precision@K y Recall@K promedio por usuario."""
    user_preds = {}
    for uid, actual, pred in zip(user_ids, actuals, preds):
        user_preds.setdefault(uid, []).append((pred, actual))

    precisions, recalls = [], []
    for uid, ratings in user_preds.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        n_rel_topk = sum(1 for _, a in top_k if a >= threshold)
        n_rel_total = sum(1 for _, a in ratings if a >= threshold)
        precisions.append(n_rel_topk / k)
        recalls.append(n_rel_topk / n_rel_total if n_rel_total > 0 else 0.0)

    return np.mean(precisions), np.mean(recalls)


def show_comparison_table(rmse_user, mae_user, rmse_item, mae_item, rmse_svd, mae_svd):
    """Muestra la tabla comparativa."""
    results = pd.DataFrame({
        "Sistema": ["User-Based CF", "Item-Based CF", "SVD (k=50)"],
        "RMSE": [round(rmse_user, 4), round(rmse_item, 4), round(rmse_svd, 4)],
        "MAE": [round(mae_user, 4), round(mae_item, 4), round(mae_svd, 4)],
    })
    best_rmse = results.loc[results["RMSE"].idxmin(), "Sistema"]
    best_mae = results.loc[results["MAE"].idxmin(), "Sistema"]

    print("\n" + "═" * 55)
    print("  COMPARACIÓN DE SISTEMAS DE RECOMENDACIÓN")
    print("═" * 55)
    print(results.to_string(index=False))
    print("─" * 55)
    print(f"  Mejor RMSE: {best_rmse}")
    print(f"  Mejor MAE:  {best_mae}")
    print("═" * 55)
    return results


def plot_rmse_mae(results):
    """Gráfico de barras RMSE y MAE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    sistemas = results["Sistema"].tolist()

    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        bars = ax.bar(sistemas, results[metric], color=colors, edgecolor="white", width=0.6)
        ax.set_title(f"{metric} por Sistema de Recomendación", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{metric} (menor es mejor)", fontsize=11)
        ax.set_ylim(0, max(results[metric]) * 1.25)
        for bar, val in zip(bars, results[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)
        ax.axhline(y=results[metric].min(), color="red", linestyle="--", alpha=0.5,
                   label=f"Mejor: {results[metric].min():.4f}")
        ax.legend()

    plt.suptitle("Comparación de los 3 Sistemas de Recomendación", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison_rmse_mae.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  → Guardado: {path}")


def plot_error_distributions(actuals_user, preds_user, actuals_item, preds_item, actuals_svd, preds_svd):
    """Histogramas de distribución de errores."""
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    errors_list = [
        np.array(actuals_user) - np.array(preds_user),
        np.array(actuals_item) - np.array(preds_item),
        np.array(actuals_svd) - np.array(preds_svd),
    ]
    names = ["User-Based CF", "Item-Based CF", "SVD"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, errors, name, color in zip(axes, errors_list, names, colors):
        ax.hist(errors, bins=50, color=color, edgecolor="white", alpha=0.85, density=True)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"Distribución de Errores\n{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Error (Real - Predicho)"); ax.set_ylabel("Densidad")
        ax.text(0.95, 0.95, f"μ = {errors.mean():.3f}\nσ = {errors.std():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Distribución de Errores de Predicción", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "error_distributions.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  → Guardado: {path}")


def plot_precision_recall(user_ids_test, preds_user, actuals_user, preds_item, actuals_item, preds_svd, actuals_svd):
    """Curvas de Precision@K y Recall@K."""
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    k_values = [5, 10, 15, 20]
    all_data = [
        ("User-Based CF", preds_user, actuals_user),
        ("Item-Based CF", preds_item, actuals_item),
        ("SVD", preds_svd, actuals_svd),
    ]

    prec_recall = {name: {"precision": [], "recall": []} for name, _, _ in all_data}
    for k in k_values:
        for name, preds, actuals in all_data:
            p, r = precision_recall_at_k(actuals, preds, user_ids_test[:len(preds)], k=k)
            prec_recall[name]["precision"].append(p)
            prec_recall[name]["recall"].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, color in zip([n for n, _, _ in all_data], colors):
        axes[0].plot(k_values, prec_recall[name]["precision"],
                     marker="o", linewidth=2.5, color=color, label=name, markersize=8)
        axes[1].plot(k_values, prec_recall[name]["recall"],
                     marker="s", linewidth=2.5, color=color, label=name, markersize=8)

    axes[0].set_title("Precision@K", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("K (Top-K recomendaciones)"); axes[0].set_ylabel("Precision")
    axes[0].set_xticks(k_values); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("Recall@K", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("K (Top-K recomendaciones)"); axes[1].set_ylabel("Recall")
    axes[1].set_xticks(k_values); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Precision y Recall por Sistema de Recomendación", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "precision_recall_at_k.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  → Guardado: {path}")
    return prec_recall


def plot_radar(results, prec_recall):
    """Radar chart comparativo."""
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    max_rmse = max(results["RMSE"])
    max_mae = max(results["MAE"])

    p10 = [prec_recall[n]["precision"][1] for n in ["User-Based CF", "Item-Based CF", "SVD"]]
    r10 = [prec_recall[n]["recall"][1] for n in ["User-Based CF", "Item-Based CF", "SVD"]]

    categories = ["Precisión\n(RMSE inv.)", "Error Abs.\n(MAE inv.)", "Precision@10", "Recall@10"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, (name, color) in enumerate(zip(["User-Based CF", "Item-Based CF", "SVD"], colors)):
        values = [
            1 - results.iloc[idx]["RMSE"] / (max_rmse * 1.1),
            1 - results.iloc[idx]["MAE"] / (max_mae * 1.1),
            p10[idx],
            r10[idx],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2.5, label=name, color=color, markersize=7)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Comparación Global de Sistemas", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "radar_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  → Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── PARTE 1: EDA ──
    df, movies = load_data()
    eda_basico(df)
    df = eda_visualizaciones(df)
    eda_correlaciones(df)
    eda_temporal(df)
    eda_generos(df)

    # ── PARTE 2: Evaluación de modelos ──
    train, test, movies_eval, n_users, n_items = load_train_test()
    user_item_matrix, user_item_centered, user_means = build_matrices(train, n_users, n_items)

    preds_user, actuals_user, rmse_user, mae_user = evaluate_user_cf(
        test, user_item_matrix, user_item_centered, user_means, n_users, n_items
    )
    preds_item, actuals_item, rmse_item, mae_item = evaluate_item_cf(
        test, user_item_matrix, user_item_centered, user_means, n_users, n_items
    )
    preds_svd, actuals_svd, rmse_svd, mae_svd = evaluate_svd(
        test, user_item_centered, user_means, n_users, n_items
    )

    # ── PARTE 3: Comparación y gráficos ──
    results = show_comparison_table(rmse_user, mae_user, rmse_item, mae_item, rmse_svd, mae_svd)
    plot_rmse_mae(results)
    plot_error_distributions(actuals_user, preds_user, actuals_item, preds_item, actuals_svd, preds_svd)

    user_ids_test = test["userId"].tolist()
    prec_recall = plot_precision_recall(
        user_ids_test, preds_user, actuals_user, preds_item, actuals_item, preds_svd, actuals_svd
    )
    plot_radar(results, prec_recall)

    print("\n══════════════════════════════════════════════════════════════")
    print("  ✅ COMPLETADO — Todos los gráficos generados en src/")
    print("══════════════════════════════════════════════════════════════")



if __name__ == "__main__":
    main()

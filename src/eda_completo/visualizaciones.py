"""
02 — Visualizaciones EDA
=========================
Genera todas las visualizaciones del análisis exploratorio:
  - Distribuciones (ratings, edad, género, ocupaciones, décadas, géneros)
  - Matriz de correlación
  - Análisis temporal por década
  - Análisis por género (popularidad, rating, co-ocurrencia, evolución)

Todas las imágenes se guardan en  src/images/
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
IMG_DIR = os.path.join(PROJECT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Columnas de género ───────────────────────────────────────────────────────
GENRE_COLS_ALL = [
    "unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film_Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci_Fi", "Thriller", "War", "Western",
]

GENRE_COLS_TOP = [
    "Action", "Comedy", "Drama", "Sci_Fi", "Thriller", "Horror", "Romance",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUCIONES GENERALES (8 gráficos)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_distribuciones(df):
    """Grid de 8 visualizaciones de distribuciones generales."""
    print("\n  Generando visualizaciones de distribuciones...")

    fig = plt.figure(figsize=(16, 12))

    # 1 — Distribución de Ratings
    ax1 = plt.subplot(3, 3, 1)
    df["rating"].value_counts().sort_index().plot(kind="bar", color="red", ax=ax1)
    ax1.set_title("Distribución de Ratings", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Frecuencia")

    # 2 — Distribución de Edad
    ax2 = plt.subplot(3, 3, 2)
    df["age"].hist(bins=30, color="blue", edgecolor="black", ax=ax2)
    ax2.set_title("Distribución de Edad", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Edad")
    ax2.set_ylabel("Frecuencia")

    # 3 — Rating por Género
    ax3 = plt.subplot(3, 3, 3)
    df.groupby("gender")["rating"].mean().plot(kind="bar", color="red", ax=ax3)
    ax3.set_title("Rating Promedio por Género", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Género")
    ax3.set_ylabel("Rating Promedio")
    ax3.set_xticklabels(["F", "M"], rotation=0)

    # 4 — Ratings por Año
    ax4 = plt.subplot(3, 3, 4)
    df.groupby("year").size().plot(color="blue", linewidth=2, ax=ax4)
    ax4.set_title("Nº de Ratings por Año de Película", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Año")
    ax4.set_ylabel("Nº de Ratings")

    # 5 — Top 10 Ocupaciones
    ax5 = plt.subplot(3, 3, 5)
    df["occupation"].value_counts().head(10).plot(kind="barh", color="red", ax=ax5)
    ax5.set_title("Top 10 Ocupaciones", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Frecuencia")

    # 6 — Rating por Grupo de Edad
    ax6 = plt.subplot(3, 3, 6)
    age_bins = [0, 18, 25, 35, 50, 100]
    age_labels = ["<18", "18-24", "25-34", "35-49", "50+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    df.groupby("age_group")["rating"].mean().plot(kind="bar", color="blue", ax=ax6)
    ax6.set_title("Rating Promedio por Grupo de Edad", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Grupo de Edad")
    ax6.set_ylabel("Rating Promedio")
    ax6.set_xticklabels(age_labels, rotation=45)

    # 7 — Películas por Década
    df["decade"] = (df["year"] // 10) * 10
    ax8 = plt.subplot(3, 3, 8)
    df.groupby("decade").size().plot(kind="bar", color="red", ax=ax8)
    ax8.set_title("Películas por Década", fontsize=12, fontweight="bold")
    ax8.set_xlabel("Década")
    ax8.set_ylabel("Número de Películas")

    # 8 — Top Géneros
    ax9 = plt.subplot(3, 3, 9)
    genre_counts = df[GENRE_COLS_TOP].sum().sort_values(ascending=True)
    genre_counts.plot(kind="barh", color="blue", ax=ax9)
    ax9.set_title("Top Géneros", fontsize=12, fontweight="bold")
    ax9.set_xlabel("Número de Películas")

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "eda_visualizations.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CORRELACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlaciones(df):
    """Heatmap de correlaciones entre variables numéricas."""
    print("\n  Generando matriz de correlación...")
    numeric_cols = ["rating", "age", "year"]
    corr = df[numeric_cols].corr()

    print("\n── MATRIZ DE CORRELACIÓN ──")
    print(corr.to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0,
                square=True, linewidths=1, cbar=True)
    plt.title("Matriz de Correlación", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "correlation_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════

def plot_temporal(df):
    """Análisis por década: películas, ratings, rating promedio y top películas."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  ANÁLISIS TEMPORAL POR DÉCADA")
    print("══════════════════════════════════════════════════════════════")

    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    movies_per_decade = df.drop_duplicates("movieId").groupby("decade").size().reset_index(name="num_movies")
    ratings_per_decade = df.groupby("decade").size().reset_index(name="num_ratings")
    avg_rating_decade = df.groupby("decade")["rating"].mean().reset_index(name="avg_rating")
    decade_summary = (
        movies_per_decade
        .merge(ratings_per_decade, on="decade")
        .merge(avg_rating_decade, on="decade")
    )
    decade_summary["avg_rating"] = decade_summary["avg_rating"].round(2)
    print(decade_summary.to_string(index=False))

    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(decade_summary["decade"].astype(str), decade_summary["num_movies"],
                color="#2196F3", edgecolor="white")
    axes[0].set_title("Nº de Películas por Década", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Década")
    axes[0].set_ylabel("Cantidad")
    for i, v in enumerate(decade_summary["num_movies"]):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=9)

    axes[1].bar(decade_summary["decade"].astype(str), decade_summary["num_ratings"],
                color="#FF9800", edgecolor="white")
    axes[1].set_title("Nº de Ratings por Década", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Década")
    axes[1].set_ylabel("Cantidad")
    for i, v in enumerate(decade_summary["num_ratings"]):
        axes[1].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold", fontsize=9)

    axes[2].plot(decade_summary["decade"].astype(str), decade_summary["avg_rating"],
                 marker="o", linewidth=2.5, color="#4CAF50", markersize=8)
    axes[2].set_title("Rating Promedio por Década", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Década")
    axes[2].set_ylabel("Rating Promedio")
    axes[2].set_ylim(2.5, 4.5)
    axes[2].axhline(y=df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                     label=f'Media global: {df["rating"].mean():.2f}')
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "temporal_decades.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")

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


# ═══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS POR GÉNERO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_generos(df):
    """Análisis completo por género: popularidad, rating, co-ocurrencia, evolución."""
    print("\n══════════════════════════════════════════════════════════════")
    print("  ANÁLISIS POR GÉNERO DE PELÍCULA")
    print("══════════════════════════════════════════════════════════════")

    # Estadísticas por género
    genre_stats = []
    for g in GENRE_COLS_ALL:
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

    # ── Popularidad y Rating por género ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    genre_sorted = genre_df.sort_values("num_ratings")
    colors_pop = plt.cm.Blues(np.linspace(0.3, 0.9, len(genre_sorted)))
    axes[0].barh(genre_sorted["genre"], genre_sorted["num_ratings"],
                 color=colors_pop, edgecolor="white")
    axes[0].set_title("Popularidad por Género (Nº de Ratings)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Cantidad de Ratings")
    for i, v in enumerate(genre_sorted["num_ratings"]):
        axes[0].text(v + 200, i, f"{v:,}", va="center", fontsize=8)

    genre_sorted_rating = genre_df.sort_values("avg_rating")
    colors_rat = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(genre_sorted_rating)))
    axes[1].barh(genre_sorted_rating["genre"], genre_sorted_rating["avg_rating"],
                 color=colors_rat, edgecolor="white")
    axes[1].set_title("Rating Promedio por Género", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Rating Promedio")
    axes[1].set_xlim(2.5, 4.5)
    axes[1].axvline(x=df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                     label=f'Media global: {df["rating"].mean():.2f}')
    axes[1].legend()
    for i, v in enumerate(genre_sorted_rating["avg_rating"]):
        axes[1].text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "genre_popularity_rating.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")

    # ── Co-ocurrencia de géneros ──
    movies_unique = df.drop_duplicates("movieId")
    genre_matrix = movies_unique[GENRE_COLS_ALL].astype(int)
    co_occurrence = genre_matrix.T.dot(genre_matrix)
    co_vals = co_occurrence.values.copy()
    np.fill_diagonal(co_vals, 0)
    co_occurrence = pd.DataFrame(co_vals, index=co_occurrence.index, columns=co_occurrence.columns)

    plt.figure(figsize=(12, 9))
    sns.heatmap(co_occurrence, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=GENRE_COLS_ALL, yticklabels=GENRE_COLS_ALL,
                linewidths=0.5, linecolor="white")
    plt.title("Co-ocurrencia de Géneros en Películas", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "genre_cooccurrence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")

    # ── Evolución de géneros por década ──
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    genre_decade = []
    for g in GENRE_COLS_ALL:
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
        plt.plot(subset["decade"].astype(str), subset["avg_rating"],
                 marker="o", linewidth=2, label=genre)
    plt.title("Evolución del Rating Promedio por Género (Top 5)", fontsize=13, fontweight="bold")
    plt.xlabel("Década")
    plt.ylabel("Rating Promedio")
    plt.legend(title="Género", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "genre_evolution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")

    return genre_df


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run(df):
    """Ejecuta todas las visualizaciones EDA."""
    df = plot_distribuciones(df)
    plot_correlaciones(df)
    plot_temporal(df)
    plot_generos(df)
    return df


if __name__ == "__main__":
    from carga_y_limpieza import load_data
    df, _ = load_data()
    run(df)

"""
03 — Evaluación de Modelos de Recomendación
=============================================
Evalúa 3 sistemas de recomendación sobre MovieLens 100K:
  1. User-Based Collaborative Filtering
  2. Item-Based Collaborative Filtering
  3. SVD (Matrix Factorization)

Métricas: RMSE, MAE, Precision@K, Recall@K.
Genera gráficos comparativos en  src/images/
"""

import os
import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "..", "data", "movielens.db")
IMG_DIR = os.path.join(PROJECT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DE SPLITS
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MATRICES
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  SISTEMA 1 — USER-BASED CF
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_user_cf(test, user_item_matrix, user_item_centered, user_means,
                     n_users, n_items, k=30):
    """Evalúa User-Based Collaborative Filtering."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SISTEMA 2 — ITEM-BASED CF
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_item_cf(test, user_item_matrix, user_item_centered, user_means,
                     n_users, n_items, k=30):
    """Evalúa Item-Based Collaborative Filtering."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SISTEMA 3 — SVD
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_svd(test, user_item_centered, user_means, n_users, n_items,
                 n_factors=50):
    """Evalúa SVD (Matrix Factorization)."""
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
#  MÉTRICAS DE RANKING
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


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLA COMPARATIVA Y GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

def show_comparison_table(rmse_user, mae_user, rmse_item, mae_item,
                          rmse_svd, mae_svd):
    """Muestra la tabla comparativa de los 3 sistemas."""
    results = pd.DataFrame({
        "Sistema": ["User-Based CF", "Item-Based CF", "SVD (k=50)"],
        "RMSE": [round(rmse_user, 4), round(rmse_item, 4), round(rmse_svd, 4)],
        "MAE": [round(mae_user, 4), round(mae_item, 4), round(mae_svd, 4)],
    })
    best_rmse = results.loc[results["RMSE"].idxmin(), "Sistema"]
    best_mae = results.loc[results["MAE"].idxmin(), "Sistema"]

    print("\n" + "=" * 55)
    print("  COMPARACIÓN DE SISTEMAS DE RECOMENDACIÓN")
    print("=" * 55)
    print(results.to_string(index=False))
    print("-" * 55)
    print(f"  Mejor RMSE: {best_rmse}")
    print(f"  Mejor MAE:  {best_mae}")
    print("=" * 55)
    return results


def plot_rmse_mae(results):
    """Gráfico de barras RMSE y MAE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    sistemas = results["Sistema"].tolist()

    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        bars = ax.bar(sistemas, results[metric], color=colors,
                      edgecolor="white", width=0.6)
        ax.set_title(f"{metric} por Sistema de Recomendación",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{metric} (menor es mejor)", fontsize=11)
        ax.set_ylim(0, max(results[metric]) * 1.25)
        for bar, val in zip(bars, results[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)
        ax.axhline(y=results[metric].min(), color="red", linestyle="--",
                   alpha=0.5, label=f"Mejor: {results[metric].min():.4f}")
        ax.legend()

    plt.suptitle("Comparación de los 3 Sistemas de Recomendación",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "model_comparison_rmse_mae.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")


def plot_error_distributions(actuals_user, preds_user,
                              actuals_item, preds_item,
                              actuals_svd, preds_svd):
    """Histogramas de distribución de errores por sistema."""
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    errors_list = [
        np.array(actuals_user) - np.array(preds_user),
        np.array(actuals_item) - np.array(preds_item),
        np.array(actuals_svd) - np.array(preds_svd),
    ]
    names = ["User-Based CF", "Item-Based CF", "SVD"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, errors, name, color in zip(axes, errors_list, names, colors):
        ax.hist(errors, bins=50, color=color, edgecolor="white",
                alpha=0.85, density=True)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"Distribución de Errores\n{name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Error (Real - Predicho)")
        ax.set_ylabel("Densidad")
        ax.text(0.95, 0.95,
                f"\u03bc = {errors.mean():.3f}\n\u03c3 = {errors.std():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Distribución de Errores de Predicción",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "error_distributions.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")


def plot_precision_recall(user_ids_test,
                           preds_user, actuals_user,
                           preds_item, actuals_item,
                           preds_svd, actuals_svd):
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
            p, r = precision_recall_at_k(actuals, preds,
                                         user_ids_test[:len(preds)], k=k)
            prec_recall[name]["precision"].append(p)
            prec_recall[name]["recall"].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, color in zip([n for n, _, _ in all_data], colors):
        axes[0].plot(k_values, prec_recall[name]["precision"],
                     marker="o", linewidth=2.5, color=color,
                     label=name, markersize=8)
        axes[1].plot(k_values, prec_recall[name]["recall"],
                     marker="s", linewidth=2.5, color=color,
                     label=name, markersize=8)

    axes[0].set_title("Precision@K", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("K (Top-K recomendaciones)")
    axes[0].set_ylabel("Precision")
    axes[0].set_xticks(k_values)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Recall@K", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("K (Top-K recomendaciones)")
    axes[1].set_ylabel("Recall")
    axes[1].set_xticks(k_values)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Precision y Recall por Sistema de Recomendación",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "precision_recall_at_k.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")
    return prec_recall


def plot_radar(results, prec_recall):
    """Radar chart comparativo de los 3 sistemas."""
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    max_rmse = max(results["RMSE"])
    max_mae = max(results["MAE"])

    p10 = [prec_recall[n]["precision"][1]
           for n in ["User-Based CF", "Item-Based CF", "SVD"]]
    r10 = [prec_recall[n]["recall"][1]
           for n in ["User-Based CF", "Item-Based CF", "SVD"]]

    categories = ["Precisión\n(RMSE inv.)", "Error Abs.\n(MAE inv.)",
                   "Precision@10", "Recall@10"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, (name, color) in enumerate(
            zip(["User-Based CF", "Item-Based CF", "SVD"], colors)):
        values = [
            1 - results.iloc[idx]["RMSE"] / (max_rmse * 1.1),
            1 - results.iloc[idx]["MAE"] / (max_mae * 1.1),
            p10[idx],
            r10[idx],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2.5, label=name,
                color=color, markersize=7)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Comparación Global de Sistemas",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "radar_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    """Ejecuta la evaluación completa de los 3 sistemas."""
    train, test, movies_eval, n_users, n_items = load_train_test()
    user_item_matrix, user_item_centered, user_means = build_matrices(
        train, n_users, n_items
    )

    preds_user, actuals_user, rmse_user, mae_user = evaluate_user_cf(
        test, user_item_matrix, user_item_centered, user_means, n_users, n_items
    )
    preds_item, actuals_item, rmse_item, mae_item = evaluate_item_cf(
        test, user_item_matrix, user_item_centered, user_means, n_users, n_items
    )
    preds_svd, actuals_svd, rmse_svd, mae_svd = evaluate_svd(
        test, user_item_centered, user_means, n_users, n_items
    )

    results = show_comparison_table(
        rmse_user, mae_user, rmse_item, mae_item, rmse_svd, mae_svd
    )
    plot_rmse_mae(results)
    plot_error_distributions(
        actuals_user, preds_user, actuals_item, preds_item, actuals_svd, preds_svd
    )

    user_ids_test = test["userId"].tolist()
    prec_recall = plot_precision_recall(
        user_ids_test, preds_user, actuals_user,
        preds_item, actuals_item, preds_svd, actuals_svd
    )
    plot_radar(results, prec_recall)

    print("\n  Evaluación de modelos completada.")


if __name__ == "__main__":
    run()

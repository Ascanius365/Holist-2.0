"""
Clustering comparison using sentence-transformers/all-MiniLM-L6-v2:
1) Without Instruction
2) With API Custom Instruction
3) With Targeted In-Text Context (Food Chain vs. Rest)
"""

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Configuration ─────────────────────────────────────────────────────────────

# Umstellung auf das kompakte MiniLM-Modell
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://api.deepinfra.com/v1/inference/{MODEL}"
TOKEN = os.environ.get("DEEPINFRA_TOKEN", "Gb3Id0aWvdB9vlE6juE42W7h78dGPpfj")

# ── Items ─────────────────────────────────────────────────────────────────────

ITEMS = [
    # Food Chain & Production
    "baking bread",
    "cooking a soup",
    "fishing",
    "harvest grain",
    "beekeeping",
    "milking cows",

    # Heavy Industry / Mining / Others
    "mine iron",
    "quarry stone",
    "poisonous mushrooms",
    "bake clay bricks"
]

# Kontext für Versuch 2 (Als API-Parameter)
INSTRUCTION_V2 = "Which of the activities are part of the economic chain involved in getting from the harvest and food acquisition to the finished dish?"

EXPECTED = {
    "baking bread": "Food Chain",
    "cooking a soup": "Food Chain",
    "fishing": "Food Chain",
    "harvest grain": "Food Chain",
    "beekeeping": "Food Chain",
    "milking cows": "Food Chain",
    "mine iron": "Other",
    "quarry stone": "Other",
    "poisonous mushrooms": "Other",
    "bake clay bricks": "Other"
}


# ── API ───────────────────────────────────────────────────────────────────────

def embed(texts: list[str], custom_instruction: str = "") -> np.ndarray:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }

    payload = {
        "inputs": texts,
        "normalize": True,
    }

    # Wir senden es mit, auch wenn MiniLM den Parameter nativ ignoriert
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction

    r = requests.post(API_URL, headers=headers, json=payload)
    r.raise_for_status()

    response_data = r.json()
    return np.array(response_data["embeddings"])


# ── Plotting Helper ───────────────────────────────────────────────────────────

COLORS_2 = ["#4C9BE8", "#E8824C"]  # Food Chain / Others
COLORS_3 = ["#4C9BE8", "#E8824C", "#5DBF6A"]  # 3-Way Splitting
MARKERS = ["o", "s", "^", "D", "v", "<", ">"]  # Varied markers for items


def plot_cluster(ax, embeddings_2d, labels, cluster_ids, n_clusters,
                 title, colors):
    """Draw a cluster scatter plot on a given axis."""
    palette = colors[:n_clusters]

    for idx, (label, cid) in enumerate(zip(labels, cluster_ids)):
        x, y = embeddings_2d[idx]
        color = palette[cid]
        marker = MARKERS[idx % len(MARKERS)]
        ax.scatter(x, y, c=color, marker=marker, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(label, (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color="#333333")

    # Mark cluster centres
    for cid in range(n_clusters):
        mask = cluster_ids == cid
        center = embeddings_2d[mask].mean(axis=0)
        ax.scatter(*center, c=palette[cid], marker="x", s=200,
                   linewidths=2.5, zorder=4)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("PCA Dim 1", fontsize=9)
    ax.set_ylabel("PCA Dim 2", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_facecolor("#F8F9FA")


def cluster_and_plot():
    print(f"Model: {MODEL}\n")

    # ── V1) Without instruction → 2 clusters ──────────────────────────────────
    print("Step 1: Embeddings without instruction (V1) ...")
    emb_without = embed(ITEMS)

    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    ids_without = km2.fit_predict(emb_without)

    pca = PCA(n_components=2, random_state=42)
    emb2d_without = pca.fit_transform(emb_without)

    # ── V2) With API Instruction → 3 clusters ─────────────────────────────────
    print("Step 2: Embeddings with API custom_instruction (V2) ...")
    emb_with_inst = embed(ITEMS, custom_instruction=INSTRUCTION_V2)

    km3_v2 = KMeans(n_clusters=3, random_state=42, n_init=10)
    ids_with_inst = km3_v2.fit_predict(emb_with_inst)

    pca2 = PCA(n_components=2, random_state=42)
    emb2d_v2 = pca2.fit_transform(emb_with_inst)

    # ── V3) With Targeted In-Text Context → 3 clusters ────────────────────────
    print("Step 3: Embeddings with targeted Context inside the Text string (V3) ...")

    items_with_context = []
    for item in ITEMS:
        if item in ["baking bread", "cooking a soup", "fishing", "harvest grain", "beekeeping", "milking cows"]:
            # Gezielter In-Text Kontext für Nahrungskette
            items_with_context.append(f"{item} harvesting, cooking")
        else:
            # Die restlichen Tätigkeiten bleiben roh/unberührt
            items_with_context.append(item)

    emb_with_context = embed(items_with_context)

    km3_v3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    ids_context = km3_v3.fit_predict(emb_with_context)

    pca3 = PCA(n_components=2, random_state=42)
    emb2d_v3 = pca3.fit_transform(emb_with_context)

    print("\nStep 3 Final Strings sent to model:")
    for original, final_str in zip(ITEMS, items_with_context):
        print(f"  -> '{final_str}'")

    print("\nStep 3 Cluster assignments:")
    for label, cid in zip(ITEMS, ids_context):
        print(f"    [{cid}] {label:<20} (expected category: {EXPECTED[label]})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating 3-way comparison plot ...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"{MODEL} · 3-Way Embedding Paradigm Comparison (Economic Activities)",
        fontsize=12, fontweight="bold", y=1.03
    )

    # Left: V1 (No instructions)
    plot_cluster(
        ax1, emb2d_without, ITEMS, ids_without, 2,
        "V1: Without Instruction\n→ 2 clusters: Raw Semantic Similarity",
        COLORS_2,
    )
    legend_v1 = [mpatches.Patch(color=COLORS_2[i], label=f"Cluster {i}") for i in range(2)]
    ax1.legend(handles=legend_v1, fontsize=8, loc="lower right")

    # Middle: V2 (API custom_instruction)
    plot_cluster(
        ax2, emb2d_v2, ITEMS, ids_with_inst, 3,
        "V2: API custom_instruction\n→ 3 clusters (Expect no instruction shift on MiniLM)",
        COLORS_3,
    )
    legend_v2 = [mpatches.Patch(color=COLORS_3[i], label=f"Cluster {i}") for i in range(3)]
    ax2.legend(handles=legend_v2, fontsize=8, loc="lower right")

    # Right: V3 (Targeted Context Injection)
    plot_cluster(
        ax3, emb2d_v3, ITEMS, ids_context, 3,
        "V3: Targeted Context Injection\n→ 3 clusters (Forced string tokens 'harvesting, cooking')",
        COLORS_3,
    )
    legend_v3 = [mpatches.Patch(color=COLORS_3[i], label=f"Cluster {i}") for i in range(3)]
    ax3.legend(handles=legend_v3, fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = "minilm_activities_cluster_3_steps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    cluster_and_plot()
import os
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import umap

# ── Farbpalette für Merge-Gruppen ────────────────────────────────────────────
# Genug Farben für bis zu 10 gleichzeitige aktive Cluster-Gruppen.
# Grau ist reserviert für unveränderte Cluster (keine neuen Fakten).
GROUP_COLORS = [
    '#e6194b',  # Rot
    '#3cb44b',  # Grün
    '#4363d8',  # Blau
    '#f58231',  # Orange
    '#911eb4',  # Lila
    '#42d4f4',  # Cyan
    '#f032e6',  # Magenta
    '#bfef45',  # Lime
    '#fabed4',  # Pink
    '#469990',  # Teal
]
UNCHANGED_COLOR = '#ced4da'   # Grau für unveränderte Cluster
UNCHANGED_EDGE  = '#6c757d'


def truncate_text(text, max_len=200):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _build_group_color_map(labels, num_facts, cluster_summaries, active_cluster_ids):
    """
    Ordnet jedem KMeans-Cluster-Label, das eine echte Merge-Gruppe bildet,
    eine eindeutige Gruppenfarbe zu.

    Eine Merge-Gruppe ist "aktiv" (bekommt eine Farbe), wenn das Label
    mindestens 2 Items enthält — egal ob das Fakten, alte Cluster-Summaries,
    oder eine Mischung aus beidem sind. Das deckt drei Fälle ab:
      • Fakt(e) + alter Cluster werden zusammengeführt
      • mehrere alte Cluster werden ohne neue Fakten zusammengeführt
      • mehrere Fakten bilden gemeinsam einen neuen Cluster
    Ein Label mit nur einem einzigen Item (ein isolierter alter Cluster ohne
    Partner) gilt als unverändert und bleibt grau.

    Gibt zurück:
        label_to_color   – dict: kmeans-label → Farbe (nur für aktive Label)
        unchanged_labels – set: kmeans-label ohne Merge-Partner
    """
    label_counts = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    active_labels = {lbl for lbl, count in label_counts.items() if count > 1}

    label_to_color = {}
    for color_idx, lbl in enumerate(sorted(active_labels)):
        label_to_color[lbl] = GROUP_COLORS[color_idx % len(GROUP_COLORS)]

    unchanged_labels = set(np.unique(labels)) - active_labels
    return label_to_color, unchanged_labels


def plot_memory_snapshot(
    state,
    labels,                        # np.array – KMeans-Labels für [Fakten | alte Cluster-Summaries]
    bot_name
):
    """
    Visualisiert den Gedächtnis-Zustand vor oder nach der Konsolidierung.

    Before-Plot
    -----------
    • Alle Fakten und die alten Cluster-Summaries, die mit ihnen zusammen-
      geführt werden, bekommen dieselbe Gruppenfarbe.
    • Alte Cluster-Summaries, die in keinem Merge-Cluster landen
      (keine neuen Fakten dabei), erscheinen in Grau.

    After-Plot
    ----------
    • Neu entstandene (konsolidierte) Cluster-Summaries erhalten die Farbe
      der Gruppe, aus der sie entstanden sind.
    • Unveränderte Cluster-Summaries (consolidated=False) bleiben Grau.
    """

    bot_base_dir = f"bots/{bot_name}"
    unified_memory_path=f"{bot_base_dir}/database/unified_memory.pkl"
    output_dir = f"{bot_base_dir}/Memory_History"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename    = os.path.join(output_dir, f"memory_snapshot_{timestamp}_{state}.png")

    # ── Daten laden ──────────────────────────────────────────────────────────
    if not os.path.exists(unified_memory_path):
        print(f"❌ Analyse-Fehler: {unified_memory_path} existiert nicht.")
        return

    with open(unified_memory_path, 'rb') as f:
        unified_memory = pickle.load(f)

    metadata         = unified_memory.get('metadata', {})
    cluster_summaries = unified_memory.get('cluster_summaries', [])

    # Im Before-State stehen noch *alle originalen* Cluster-Summaries im Speicher,
    # d.h. die, die direkt aus unified_memory kommen, bevor sie überschrieben wurden.
    # Die Fakten sind ebenfalls noch vorhanden.
    facts_embeddings = np.array(unified_memory.get('facts', []))

    # Cluster-Embeddings (nur valide)
    cluster_embeddings = []
    valid_summaries    = []
    for s in cluster_summaries:
        if s.get('embedding') is not None:
            cluster_embeddings.append(s['embedding'])
            valid_summaries.append(s)
    cluster_embeddings = np.array(cluster_embeddings) if cluster_embeddings else np.array([])

    num_facts    = len(facts_embeddings)
    num_clusters = len(cluster_embeddings)

    if num_facts == 0 and num_clusters == 0:
        print("⚠️ Keine Fakten oder Cluster zum Visualisieren vorhanden.")
        return

    # ── Gemeinsame Embedding-Matrix für UMAP ─────────────────────────────────
    if num_facts > 0 and num_clusters > 0:
        X = np.vstack([facts_embeddings, cluster_embeddings])
    elif num_facts > 0:
        X = facts_embeddings
    else:
        X = cluster_embeddings

    if len(X) < 2:
        X_2d = np.zeros((len(X), 2))
    else:
        reducer = umap.UMAP(
            n_neighbors=min(5, len(X) - 1),
            min_dist=0.005,
            metric='cosine',
            random_state=42,
        )
        X_2d = reducer.fit_transform(X)

    # ── Gruppen-Farbzuordnung aus den KMeans-Labels ───────────────────────────
    # labels hat dieselbe Länge wie X, also [Fakten | alte Cluster-Summaries].
    # Aktive Gruppen = Labels die mindestens einen Fakt enthalten.
    if labels is not None and len(labels) == len(X):
        label_to_color, unchanged_labels = _build_group_color_map(
            labels, num_facts, valid_summaries, active_cluster_ids=None
        )
    else:
        # Fallback: kein Label-Info → alle grau
        label_to_color    = {}
        unchanged_labels  = set()

    # ── Hilfsfunktionen ───────────────────────────────────────────────────────
    def fact_color(fact_idx):
        """Gibt die Gruppenfarbe für einen Fakt zurück (anhand seines KMeans-Labels)."""
        if labels is not None and fact_idx < len(labels):
            lbl = labels[fact_idx]
            return label_to_color.get(lbl, UNCHANGED_COLOR)
        return UNCHANGED_COLOR

    def cluster_color_before(cluster_pos_in_X):
        """
        Vor der Konsolidierung: Farbe eines alten Cluster-Summaries.
        Liegt er in einer Gruppe mit neuen Fakten → Gruppenfarbe, sonst Grau.
        """
        if labels is not None and cluster_pos_in_X < len(labels):
            lbl = labels[cluster_pos_in_X]
            return label_to_color.get(lbl, UNCHANGED_COLOR)
        return UNCHANGED_COLOR

    def cluster_color_after(summary):
        """
        Nach der Konsolidierung: Neue Summaries (consolidated=True) bekommen
        die Gruppenfarbe; unveränderte bleiben Grau.
        Die Gruppenfarbe wird über das rohe KMeans-Label dieses Laufs
        ('kmeans_label') ermittelt — NICHT über 'cluster_id', da cluster_id
        bei behaltenen alten Summaries aus einem früheren Lauf stammen kann
        und mit der aktuellen Lauf-Nummerierung kollidieren würde.
        """
        if not summary.get('consolidated', False):
            return UNCHANGED_COLOR
        lbl = summary.get('kmeans_label')
        if lbl is not None:
            color = label_to_color.get(lbl)
            if color is not None:
                return color
            return GROUP_COLORS[int(lbl) % len(GROUP_COLORS)]
        # Alte Daten ohne 'kmeans_label' (vor diesem Fix gespeichert): jeder
        # solchen Summary eine eigene, stabile Farbe geben statt alle gleich
        # einzufärben (sonst sehen unzusammenhängende Cluster fälschlich
        # gleich aus, wie es vor dem Fix passierte).
        key = summary.get('cluster_id', summary.get('description', ''))
        fallback_idx = abs(hash(str(key))) % len(GROUP_COLORS)
        return GROUP_COLORS[fallback_idx]

    # ── Plot aufbauen ─────────────────────────────────────────────────────────
    plt.figure(figsize=(15, 10))
    sns.set_theme(style="whitegrid")

    start_cluster_idx = num_facts  # Offset: ab hier liegen Cluster-Punkte in X_2d

    # ── 1. Fakten plotten ─────────────────────────────────────────────────────
    if num_facts > 0:
        if state == "before":
            # Jeder Fakt bekommt seine Gruppenfarbe
            for i in range(num_facts):
                color = fact_color(i)
                plt.scatter(
                    X_2d[i, 0], X_2d[i, 1],
                    color=color, marker="o", s=90, alpha=0.85,
                    edgecolor='k', linewidth=0.7, zorder=3,
                )
                meta_entry = metadata.get(i, {})
                text = truncate_text(meta_entry.get('value', ''))
                if text:
                    plt.annotate(
                        text, (X_2d[i, 0], X_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7, color='navy',
                    )
        else:
            # Im After-Plot: Fakten sind bereits in Cluster-Summaries aufgegangen.
            # Falls noch unkomprimierte Fakten übrig sind (nicht alle geclustert),
            # zeigen wir sie in hellgrau an, um die "Restmenge" sichtbar zu machen.
            for i in range(num_facts):
                plt.scatter(
                    X_2d[i, 0], X_2d[i, 1],
                    color='#adb5bd', marker="o", s=70, alpha=0.5,
                    edgecolor='k', linewidth=0.5, zorder=2,
                )

    # ── 2. Cluster-Summaries plotten ─────────────────────────────────────────
    if num_clusters > 0:
        for i, summary in enumerate(valid_summaries):
            idx_in_X  = start_cluster_idx + i
            x, y      = X_2d[idx_in_X, 0], X_2d[idx_in_X, 1]

            if state == "before":
                color     = cluster_color_before(idx_in_X)
                is_active = (color != UNCHANGED_COLOR)
            else:
                color     = cluster_color_after(summary)
                is_active = summary.get('consolidated', False)

            edgecolor  = 'black'       if is_active else UNCHANGED_EDGE
            linewidth  = 1.8           if is_active else 1.0
            alpha      = 0.95          if is_active else 0.55
            marker_s   = 220           if is_active else 160

            plt.scatter(
                x, y,
                color=color, marker="s", s=marker_s, alpha=alpha,
                edgecolor=edgecolor, linewidth=linewidth, zorder=4,
            )

            # Textannotation
            text       = truncate_text(summary.get('description', 'Summary'), max_len=70)
            box_color  = 'yellow'  if is_active else '#f8f9fa'
            box_alpha  = 0.40      if is_active else 0.15
            text_color = 'black'   if is_active else '#6c757d'
            edge_c     = 'gray'    if is_active else '#dee2e6'
            line_c     = 'gray'    if is_active else '#ced4da'
            fw         = 'bold'    if is_active else 'normal'

            plt.annotate(
                text, (x, y),
                xytext=(12, 12), textcoords='offset points',
                fontsize=9, fontweight=fw, color=text_color,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    fc=box_color, alpha=box_alpha, edgecolor=edge_c,
                ),
                arrowprops=dict(
                    arrowstyle='->', connectionstyle='arc3,rad=0.1', color=line_c
                ),
                zorder=5,
            )

    # ── 3. Legende ────────────────────────────────────────────────────────────
    legend_handles = []

    # Gruppenfarben (aktive Merge-Gruppen)
    for lbl, color in sorted(label_to_color.items()):
        if state == "before":
            label_str = f"Merge-Gruppe {lbl} (Fakten + zugehörige Cluster)"
        else:
            label_str = f"Konsolidierter Cluster aus Gruppe {lbl}"
        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=label_str)
        )

    # Grau für unveränderte Cluster
    if any(
        (cluster_color_before(start_cluster_idx + i) == UNCHANGED_COLOR
         if state == "before"
         else not valid_summaries[i].get('consolidated', False))
        for i in range(num_clusters)
    ):
        legend_handles.append(
            mpatches.Patch(
                facecolor=UNCHANGED_COLOR, edgecolor=UNCHANGED_EDGE,
                label="Unveränderte Cluster (keine neuen Fakten)",
            )
        )

    # Fakten-Marker nur im Before-Plot erklären
    if state == "before" and num_facts > 0:
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='#555',
                markersize=9, label=f"Unkomprimierte Fakten ({num_facts})",
            )
        )
    elif state == "after" and num_facts > 0:
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='#adb5bd',
                markersize=9, label=f"Verbleibende unkomprimierte Fakten ({num_facts})",
            )
        )

    plt.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True, shadow=True, fontsize=10,
    )

    # ── 4. Titel & Achsenbeschriftungen ──────────────────────────────────────
    state_title    = "VOR Consolidation" if state == "before" else "NACH Consolidation"
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plt.title(
        f"Gedächtnis-Struktur von Holist 2.0 ({state_title}) — Snapshot um {current_time_str}",
        fontsize=16, fontweight='bold', pad=15,
    )
    plt.xlabel("UMAP Komponente 1 (Semantische Breite)", fontsize=11)
    plt.ylabel("UMAP Komponente 2 (Semantische Tiefe)", fontsize=11)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()
    #print(f"📊 [VERLAUF] Snapshot erfolgreich gespeichert: {filename}")
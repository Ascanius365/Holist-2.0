import sys
import asyncio
import logging
import sys
import warnings

from tqdm import tqdm
from sklearn.cluster import DBSCAN

sys.path.append(".")
from consolidation.io import read_jsonl, save_jsonl

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

import os
import pandas as pd
import pickle
import numpy as np
import hashlib
import json
import re
import ast
from datetime import datetime

import litellm
#litellm._turn_on_debug()


def fix_incomplete_json(json_str, session_id="Unbekannt"):
    # Falls es bereits ein Dictionary ist, direkt zurückgeben
    if isinstance(json_str, dict):
        return json_str

    if not isinstance(json_str, str):
        return {}

    # Gedanken-Tags und Markdown-Codeblöcke entfernen
    if "<think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    # --- SCHRITT 1: Direktes Laden versuchen ---
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # --- SCHRITT 2: Intelligentes Schließen offener Anführungszeichen & Klammern ---
    fixed_str = json_str
    in_string = False
    escape = False
    stack = []

    # Zeichen für Zeichen analysieren
    for char in fixed_str:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in ('{', '['):
                stack.append(char)
            elif char in ('}', ']'):
                if stack:
                    stack.pop()

    # Falls das LLM mitten in einem String abgebrochen ist (wie bei '"date": "')
    if in_string:
        fixed_str += '"'

    # Alle noch offenen Klammern in umgekehrter Reihenfolge schließen (z.B. erst }, dann ], dann })
    while stack:
        open_char = stack.pop()
        if open_char == '{':
            fixed_str += '}'
        elif open_char == '[':
            fixed_str += ']'

    # --- SCHRITT 3: Reparierten String testen ---
    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass

    # --- SCHRITT 4: Letzte Kommas entfernen (Trailing Commas, z.B. ,"_}) ---
    try:
        cleaned_str = re.sub(r',\s*([\]}])', r'\1', fixed_str)
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        pass

    # --- SCHRITT 5: Python AST Fallback (für falsche Single Quotes) ---
    try:
        evaluated = ast.literal_eval(json_str)
        if isinstance(evaluated, dict):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    # Wenn absolut gar nichts hilft, Fehler loggen
    print(f"⚠️  JSON für Session '{session_id}' absolut nicht reparierbar!")
    print(f"   Inhalt: {json_str[:120]}...")
    return {}


def fix_incomplete_json2(json_str):
    """Parse JSON mit Fehlerbehandlung"""
    if "<think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            try:
                return json.loads(json_str[:last_brace + 1])
            except:
                pass
    return {}


def get_fact_hash(fact_dict):
    """Generiert einen eindeutigen MD5-Hash zur Deduplizierung von atomaren Fakten."""
    fact_str = json.dumps({
        'session_id': fact_dict.get('session_id'),
        'key': fact_dict.get('key'),
        'value': fact_dict.get('value'),
        'date': fact_dict.get('date')
    }, sort_keys=True)
    return hashlib.md5(fact_str.encode()).hexdigest()


def update_long_term_memory(args, agent, bot_name, embed_req_q, embed_res_q, logger):
    """
    Stufe 1: Lädt neue Episoden/Einträge aus Memory/sessions.jsonl,
    extrahiert Fakten, bettet sie ein, aktualisiert das unified_memory.pkl
    und löscht erfolgreich verarbeitete Zeilen aus der Quelldatei.
    """
    input_file = f"bots/{bot_name}/memory/sessions.jsonl"
    if not os.path.exists(input_file):
        print(f"⚠️ {input_file} nicht gefunden. Überspringe Ingestion-Schritt.")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as read_error:
        print(f"❌ Fehler beim Lesen von {input_file}: {read_error}")
        return

    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    unified_file = f"{bot_base_dir}/database/unified_memory.pkl"

    # Unified Memory laden oder neu initialisieren
    if os.path.exists(unified_file):
        with open(unified_file, 'rb') as f:
            unified_memory = pickle.load(f)
        logger.info(f"✅ Unified Memory geladen ({len(unified_memory['facts'])} Fakten).")
    else:
        unified_memory = {
            'facts': [],
            'metadata': {},
            'cluster_summaries': [],
            'processed_hashes': set(),
            'version': 1
        }
        print("📝 Neues Unified Memory initialisiert.")

    if 'processed_hashes' not in unified_memory:
        unified_memory['processed_hashes'] = set()

    remaining_lines = []
    info_list = []

    # Zeilen einzeln verarbeiten und validieren
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        data = None
        extra_text = ""

        try:
            # Versuch 1: Normales Laden
            data = json.loads(line)
        except json.JSONDecodeError as e:
            # ==================== NEU: AUTOMATISCHER SPLIT BEI EXTRA DATA ====================
            # Falls gültiges JSON + Freitext in einer Zeile stehen, trennen wir sie hier sauber auf.
            if "Extra data" in e.msg:
                try:
                    json_part = line[:e.pos].strip()
                    extra_text = line[e.pos:].strip()
                    data = json.loads(json_part)  # Lädt die Session-Metadaten
                except Exception:
                    data = None

            # Falls es reiner Freitext ohne jegliches JSON-Format war (z.B. Zeile startet nicht mit '{')
            if not data:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("{"):
                    data = {
                        "session_id": f"mc_{i}",
                        "session_date": "2024-01-01",
                        "text": clean_line
                    }
                else:
                    # Wirklich kaputtes JSON
                    print(f"⚠️ Ungültiges JSON in Zeile {i + 1}: {str(e)[:80]}")
                    remaining_lines.append(line)
                    continue
            # ==================================================================================

        try:
            if not isinstance(data, dict):
                print(f"⚠️ Zeile {i + 1} ist kein gültiges JSON-Objekt (Überspringe).")
                remaining_lines.append(line)
                continue

            # Text aus dem JSON holen und den extrahierten Freitext anhängen
            response = data.get("response") or data.get("text", "")
            if extra_text:
                if response:
                    response = f"{response}\n{extra_text}".strip()
                else:
                    response = extra_text

            if not response:
                # Zeile enthält keinen Inhalt, gilt als verarbeitet
                continue

            # Falls die Antwort bereits ein Dict (JSON) ist, direkt nutzen
            if isinstance(response, dict):
                info = response
            elif isinstance(response, str) and response.strip():
                clean_response = response.strip()

                # Nur wenn der Text mit '{' beginnt, versuchen wir eine JSON-Reparatur
                if clean_response.startswith("{"):
                    info = fix_incomplete_json(clean_response, data.get('session_id', 'Unbekannt'))
                else:
                    info = None  # Reiner Freitext, überspringe Reparatur-Funktion

                # TEXT-ZU-FAKT KONVERTIERUNG
                if not info:
                    info = {
                        "Episodic_Fact": [
                            {
                                "key": "Agent_Observation",
                                "value": clean_response
                            }
                        ]
                    }
            else:
                info = None

            if not info:
                remaining_lines.append(line)
                continue

            # Extraktion aller Informationstypen
            for info_type, values in info.items():
                if not isinstance(values, list):
                    continue
                for v in values:
                    if not v.get("value"):
                        continue
                    info_list.append({
                        "session_id": data.get("session_id", f"mc_{i}"),
                        "text": data.get("text", response if isinstance(response, str) else ""),
                        "session_date": data.get("session_date", ""),
                        "information_type": info_type,
                        "key": v.get("key", "info"),
                        "value": v["value"],
                        "date": v.get("date", "2024-01-01"),
                        "message_id": v.get("message_id", "m0")
                    })
        except Exception as e:
            print(f"❌ Unerwarteter Fehler in Zeile {i + 1}: {e}")
            remaining_lines.append(line)

    if not info_list:
        print("✅ Keine extrahierbaren Informationen in sessions.jsonl gefunden.")
        # Quelldatei aktualisieren (bereinigen)
        with open(input_file, 'w', encoding='utf-8') as f:
            f.writelines(remaining_lines)
        return

    # Filterung auf Duplikate
    new_facts = []
    for fact in info_list:
        fact_hash = get_fact_hash(fact)
        if fact_hash not in unified_memory['processed_hashes']:
            new_facts.append(fact)

    if not new_facts:
        print("✅ Alle extrahierten Fakten existieren bereits im Langzeitgedächtnis.")
        with open(input_file, 'w', encoding='utf-8') as f:
            f.writelines(remaining_lines)
        return

    logger.info(f"➕ Generiere Embeddings für {len(new_facts)} neue Fakten...")
    info_df = pd.DataFrame(new_facts)
    text_for_embed = (info_df["key"] + ": " + info_df["value"].astype(str)).tolist()

    #all_embeddings = db.embedder.create(text_for_embed)
    all_embeddings = get_shared_embedding(text_for_embed, bot_name, embed_req_q, embed_res_q)

    # In Unified Memory einspeisen
    before_count = len(unified_memory['facts'])
    for emb, meta in zip(all_embeddings, new_facts):
        unified_memory['facts'].append(emb)
        unified_memory['metadata'][len(unified_memory['facts']) - 1] = meta
        unified_memory['processed_hashes'].add(get_fact_hash(meta))

    logger.info(f"💾 {len(all_embeddings)} neue Fakten hinzugefügt ({before_count} ➔ {len(unified_memory['facts'])} gesamt).")

    # Unified Memory wegschreiben
    with open(unified_file, 'wb') as f:
        pickle.dump(unified_memory, f)

    # Quelldatei aktualisieren (erfolgreiche Zeilen löschen)
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(remaining_lines)
    logger.info(f"🧹 {input_file} aktualisiert. {len(remaining_lines)} Zeilen verbleiben.")

    run_reasoning_consolidation(args, agent, bot_name, embed_req_q, embed_res_q, logger)


def get_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def async_batch_process_tasks_and_save(tasks, params, output_path, batch_size=128):
    for i in tqdm(range(0, len(tasks), batch_size)):
        batch_tasks = tasks[i: i + batch_size]
        batch_results = await asyncio.gather(*[asyncio.wait_for(t, timeout=60) for t in batch_tasks],
                                             return_exceptions=True)

        new_data = []
        for response, param in zip(batch_results, params[i: i + batch_size]):
            # ==================== GEÄNDERT: FEHLER NICHT MEHR VERSCHLUCKEN ====================
            if isinstance(response, Exception):
                print(f"\n❌ LiteLLM API-Fehler im Cluster {param.get('cluster_id')}: {response}")
                continue
            if response is None:
                continue
            # ==================================================================================
            if isinstance(response, Exception) or response is None:
                continue
            new_data.append({**param, **response.to_dict()})

        if os.path.exists(output_path):
            existing = read_jsonl(output_path)
            save_jsonl(existing + new_data, output_path)
        else:
            save_jsonl(new_data, output_path)


def run_reasoning_consolidation(args, agent, bot_name, embed_req_q, embed_res_q, logger):
    loop = asyncio.get_event_loop()

    # ============ ÄNDERUNG 1: Lade unified_memory.pkl ============
    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    unified_file = f"{bot_base_dir}/database/unified_memory.pkl"
    if not os.path.exists(unified_file):
        print(f"❌ Fehler: {unified_file} nicht gefunden!")
        return

    with open(unified_file, 'rb') as f:
        unified_memory = pickle.load(f)

    logger.info(f"✅ Loaded unified consolidation:")
    logger.info(f"   - Fakten: {len(unified_memory['facts'])}")
    logger.info(f"   - Cluster-Summaries: {len(unified_memory.get('cluster_summaries', []))}")

    # ============ ÄNDERUNG 2: Erstelle kombinierte Embedding-Matrix ============
    # Fakten + alte Cluster-Summaries
    facts_embeddings = np.array(unified_memory['facts']) if unified_memory['facts'] else np.array([])
    cluster_summaries = unified_memory.get('cluster_summaries', [])
    cluster_embeddings = np.array([s['embedding'] for s in cluster_summaries]) if cluster_summaries else np.array([])

    # Kombiniere
    if len(facts_embeddings) > 0 and len(cluster_embeddings) > 0:
        X = np.vstack([facts_embeddings, cluster_embeddings])
    elif len(facts_embeddings) > 0:
        X = facts_embeddings
    elif len(cluster_embeddings) > 0:
        X = cluster_embeddings
    else:
        print("❌ Keine Fakten und keine Summaries gefunden!")
        return

    num_items = len(X)
    num_facts = len(facts_embeddings)
    fact_offset = num_facts  # Index ab dem Summaries beginnen

    print(f"{bot_name}: 📊 Clustering {num_items} items ({num_facts} facts + {len(cluster_summaries)} summaries)...")
    """
    # ============ ÄNDERUNG 3: Clustering ============
    raw_target = num_items // args.compression_factor
    clamped_target = max(2, min(10, raw_target))
    target_clusters = min(clamped_target, num_items)"""

    # ============ ÄNDERUNG 3: Dichte-basiertes Clustering via DBSCAN (min_samples=1) ============
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # eps=0.15 verlangt weiterhin eine Kosinus-Ähnlichkeit von 0.85 für Fusionen
    eps_value = 0.04

    if num_items >= 1:
        # min_samples=1 erlaubt Single-Element-Cluster direkt nativ. Es gibt KEIN Rauschen (-1) mehr!
        dbscan = DBSCAN(eps=eps_value, min_samples=1, metric='cosine')
        labels = dbscan.fit_predict(X_norm)

        unique_labels = [int(l) for l in np.unique(labels)]
        target_clusters = len(unique_labels)
        print(f"{bot_name}: 📊 DBSCAN (min_samples=1) abgeschlossen. Cluster gesamt: {target_clusters}")
    else:
        labels = np.array([])
        unique_labels = []
        target_clusters = 0

    # ============ Analysis before consolidation
    try:
        from consolidation.analysis import plot_memory_snapshot
        plot_memory_snapshot(bot_name = bot_name,
                             state="before", labels=labels)
    except Exception as plot_err:
        print(f"⚠️  Konnte Vor-Consolidation-Snapshot nicht erstellen: {plot_err}")


    all_keys = []
    all_tasks = []

    # ============ ÄNDERUNG 4: Für jeden Cluster entscheiden: neue Summary oder alte behalten ============
    clusters_info = []
    new_summaries_to_create = []  # Nur Cluster mit neuen Fakten
    old_summaries_to_keep = []  # Cluster nur mit alten Summaries

    run_id = pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")

    for c_id in range(target_clusters):
        cluster_indices = np.where(labels == c_id)[0]
        cluster_indices_list = cluster_indices.tolist()

        # Prüfe ob neue Fakten in diesem Cluster sind
        has_new_facts = any(idx < fact_offset for idx in cluster_indices_list)
        num_items_in_cluster = len(cluster_indices_list)
        has_old_summary = any(idx >= fact_offset for idx in cluster_indices_list)

        print(
            f"   Cluster {c_id}: {len(cluster_indices)} items ({sum(1 for i in cluster_indices_list if i < fact_offset)} facts, "
            f"{sum(1 for i in cluster_indices_list if i >= fact_offset)} summaries)")

        # ============ ÄNDERUNG 5: NUR wenn neue Fakten = neuer Reasoning-Task ============
        #if has_new_facts:
        if has_new_facts or (num_items_in_cluster > 1):
            memory_fragments_list = []

            for idx in cluster_indices_list:
                if idx < fact_offset:
                    # Neuer Fakt
                    meta = unified_memory['metadata'].get(idx, {})
                    memory_fragments_list.append(
                        f"[{meta.get('key', 'info')}, {meta.get('session_date', 'N/A')}]: "
                        f"{meta.get('role', 'user')} {meta.get('value', '')}"
                    )
                else:
                    # Alte Summary
                    summary_idx = idx - fact_offset
                    summary = cluster_summaries[summary_idx]
                    memory_fragments_list.append(
                        f"[EXISTING CLUSTER] {summary['description']}"
                    )

            # Beteiligte alte Summaries merken (für Fallback, falls der LLM-Call fehlschlägt)
            involved_old_summaries = [
                cluster_summaries[idx - fact_offset]
                for idx in cluster_indices_list if idx >= fact_offset
            ]
            involved_fact_values = [
                unified_memory['metadata'].get(idx, {}).get('value', '')
                for idx in cluster_indices_list if idx < fact_offset
            ]

            memory_fragments = "\n\n--- Nächste ---\n\n".join(memory_fragments_list)

            unique_cluster_id = f"{run_id}_{c_id}"

            cluster_info = {
                "cluster_id": unique_cluster_id,
                "kmeans_label": int(c_id),
                "fact_indices": cluster_indices_list,
                "size": len(cluster_indices),
                "has_new_facts": True,
                "involved_old_summaries": involved_old_summaries,  # NEU
                "involved_fact_values": involved_fact_values,
            }
            clusters_info.append(cluster_info)
            new_summaries_to_create.append(c_id)

            all_keys.append({
                "cluster_id": unique_cluster_id,
                "kmeans_label": int(c_id),
                "item_count": len(cluster_indices),
                "cluster_hash": hashlib.md5(str(cluster_indices).encode()).hexdigest()
            })

            all_tasks.append(agent.get_completion(
                "consolidation/reason_info.yaml",
                memory_fragments=memory_fragments,
                api_base=args.api_base
            ))
        else:
            # ============ ÄNDERUNG 6: Keine neuen Fakten = alte Summary behalten ============
            logger.info(f"   ⏭️  Cluster {c_id} hat nur alte Summaries → behalte unverändert")

            for idx in cluster_indices_list:
                summary_idx = idx - fact_offset

                old_summary = cluster_summaries[summary_idx]

                # NEU: Als nicht konsolidiert markieren
                old_summary['consolidated'] = False
                old_summary['kmeans_label'] = None

                old_summaries_to_keep.append(old_summary)

    print(f"{bot_name}: ➕ Erstelle {len(new_summaries_to_create)} neue Summaries")
    print(f"{bot_name}: ⏭️  Behalte {len(old_summaries_to_keep)} alte Summaries unverändert")

    if not all_tasks:
        print("✅ Keine neuen Fakten in Clustern → behalte alle alten Summaries")
        # NEU: Alle bestehenden auf False setzen
        for s in unified_memory.get('cluster_summaries', []):
            s['consolidated'] = False
        # Speichere unverändert
        with open(unified_file, 'wb') as f:
            pickle.dump(unified_memory, f)
        return

    # ============ ÄNDERUNG 7: Asynchrone Verarbeitung ============
    output_file = f"bots/{bot_name}/database/ours_reasoning.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)

    loop.run_until_complete(async_batch_process_tasks_and_save(
        all_tasks, all_keys, output_file, batch_size=args.batch_size
    ))

    # ============ ÄNDERUNG 8: Extrahiere neue Summaries ============
    new_summaries = old_summaries_to_keep  # Starte mit den beibehaltenen alten

    if os.path.exists(output_file):
        reasoning_results = read_jsonl(output_file)

        pending_new_summaries = []
        for result in reasoning_results:
            response_text = result.get('response', '')
            # Nutze deine passende JSON-Fix-Funktion aus dem Skript
            resp = fix_incomplete_json(response_text)

            if resp and "extended_insight" in resp:
                insights = resp.get('extended_insight', [])

                # Bilde eine lückenlose Beschreibung aus ALLEN Keys und Values des Clusters
                if insights:
                    insight_list = []
                    keys = []
                    values = []
                    for i in insights:
                        if i.get('value'):
                            key = i.get('key', '').strip()
                            value = i.get('value', '').strip()
                            date = i.get('date', 'N/A').strip()

                            # Format: "[Datum] Key: Value"
                            insight_list.append(f"[{date}] {key}: {value}".strip(": "))
                            #key_list.append(f"{key} ")
                            # In run_reasoning.py beim Erstellen der Vektoren:
                            keys.append(key)
                            values.append(value)

                    clean_keys = ", ".join(keys)
                    clean_values = " | ".join(values)

                    description = f"Minecraft Bot Memory | Topics: {clean_keys} | Context: {clean_values}"
                    #description = ". ".join(key_list) if key_list else 'Summary'
                else:
                    description = 'Summary'

                summary = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'cluster_id': result.get('cluster_id'),
                    'kmeans_label': result.get('kmeans_label'),
                    'item_count': result.get('item_count'),
                    'description': description,
                    'response': response_text,
                    'insights': insights,
                    'embedding': None,  # Wird gleich im Batch befüllt
                    'consolidated': True
                }
                pending_new_summaries.append(summary)

        # Zusammenführen mit den beibehaltenen alten Summaries
        new_summaries.extend(pending_new_summaries)

    # ============ NEU: Fallback für fehlgeschlagene Cluster ============
    succeeded_cluster_ids = {s['cluster_id'] for s in new_summaries if
                             s.get('consolidated') and s.get('cluster_id') in {c['cluster_id'] for c in clusters_info}}

    recovered_count = 0
    for c_info in clusters_info:
        if c_info['cluster_id'] in succeeded_cluster_ids:
            continue

        print(
            f"⚠️  Cluster {c_info['cluster_id']} hatte keinen gültigen LLM-Output → Fallback-Summary statt Datenverlust")

        old_summaries = c_info.get('involved_old_summaries', [])
        fact_values = c_info.get('involved_fact_values', [])

        if old_summaries:
            description = ". ".join(s.get('description', '') for s in old_summaries if s.get('description'))
            merged_insights = []
            for s in old_summaries:
                merged_insights.extend(s.get('insights', []))
        else:
            description = "; ".join(v for v in fact_values if v) or "Summary (Fallback nach LLM-Fehler)"
            merged_insights = []

        fallback_summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'cluster_id': c_info['cluster_id'],
            'kmeans_label': c_info.get('kmeans_label'),
            'item_count': c_info.get('size'),
            'description': description,
            'response': None,
            'insights': merged_insights,
            'embedding': None,
            'consolidated': True,
            'fallback': True,
        }
        new_summaries.append(fallback_summary)
        recovered_count += 1

    if recovered_count:
        print(f"{bot_name}: ♻️  {recovered_count} Cluster über Fallback gerettet (keine Daten verloren)")

    # ── NEU: Vektorisierung der Cluster UND individuellen Insights im Batch ──
    cluster_texts_to_embed = []
    cluster_indices = []

    insight_texts_to_embed = []
    insight_coordinates = []  # Speichert Tupel: (summary_index, insight_index)
    """
    # ── NEU: Vektorisierung der Cluster (via Zentroid) UND individuellen Insights im Batch ──
    insight_texts_to_embed = []
    insight_coordinates = []  # Speichert Tupel: (summary_index, insight_index)

    for s_idx, summary in enumerate(new_summaries):
        # 1. Haupt-Cluster-Embedding als mathematischen Durchschnitt (Zentroid) berechnen
        if summary.get('embedding') is None or not isinstance(summary['embedding'], np.ndarray):
            c_id = summary.get('cluster_id')
            if c_id is not None:
                # Extrahiere alle Vektoren (Fakten/alte Cluster), die diesem Cluster zugeordnet wurden
                cluster_item_embeddings = X[labels == c_id]

                if len(cluster_item_embeddings) > 0:
                    # Mathematischer Mittelwert über alle Element-Embeddings im Cluster
                    centroid = np.mean(cluster_item_embeddings, axis=0)

                    # L2-Normalisierung für korrekte Metrik bei Cosine Similarity
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm

                    summary['embedding'] = centroid
                else:
                    summary['embedding'] = None

        # 2. Jedes einzelne Insight im Cluster prüfen und für Batch-Embedding vormerken
        for i_idx, insight in enumerate(summary.get('insights', [])):
            if insight.get('embedding') is None:
                key = insight.get('key', '').strip()
                value = insight.get('value', '').strip()

                # Formatiere den Text, der semantisch repräsentiert werden soll
                insight_text = f"{key}: {value}" if key else value
                if insight_text:
                    insight_texts_to_embed.append(insight_text)
                    insight_coordinates.append((s_idx, i_idx))

    # HINWEIS: Die alte Batch-Generierung 'if cluster_texts_to_embed:' wurde komplett entfernt!

    # Batch-Generierung NUR noch für alle einzelnen Extended Insights (falls vorhanden)
    if insight_texts_to_embed:
        print(f"🔮 Generiere echte Embeddings für {len(insight_texts_to_embed)} individuelle Extended Insights...")
        i_vectors = db.embedder.create(insight_texts_to_embed, show_progress_bar=False)
        for (s_idx, i_idx), vec in zip(insight_coordinates, i_vectors):
            new_summaries[s_idx]['insights'][i_idx]['embedding'] = vec"""


    for s_idx, summary in enumerate(new_summaries):
        # 1. Haupt-Cluster-Embedding prüfen (falls neu oder unvollständig)
        if summary.get('embedding') is None or not isinstance(summary['embedding'], np.ndarray):
            cluster_texts_to_embed.append(summary['description'])
            cluster_indices.append(s_idx)

        # 2. Jedes einzelne Insight im Cluster prüfen und für Batch-Embedding vormerken
        for i_idx, insight in enumerate(summary.get('insights', [])):
            if insight.get('embedding') is None:
                key = insight.get('key', '').strip()
                value = insight.get('value', '').strip()
                inf_type = insight.get('inference_type', insight.get('type', 'info')).strip()

                # Formatiere den Text, der semantisch repräsentiert werden soll
                #insight_text = f"[{inf_type}] {key}: {value}" if key else value
                insight_text = f"{key}: {value}" if key else value
                if insight_text:
                    insight_texts_to_embed.append(insight_text)
                    insight_coordinates.append((s_idx, i_idx))

    # Batch-Generierung für Cluster-Vektoren
    if cluster_texts_to_embed:
        logger.info(f"🔮 Generiere echte Embeddings für {len(cluster_texts_to_embed)} Cluster-Zusammenfassungen...")
        #c_vectors = db.embedder.create(cluster_texts_to_embed, show_progress_bar=False)
        c_vectors = get_shared_embedding(cluster_texts_to_embed, bot_name, embed_req_q, embed_res_q)
        for s_idx, vec in zip(cluster_indices, c_vectors):
            new_summaries[s_idx]['embedding'] = vec

    # Batch-Generierung für alle einzelnen Extended Insights
    if insight_texts_to_embed:
        logger.info(f"🔮 Generiere echte Embeddings für {len(insight_texts_to_embed)} individuelle Extended Insights...")
        #i_vectors = db.embedder.create(insight_texts_to_embed, show_progress_bar=False)
        i_vectors = get_shared_embedding(insight_texts_to_embed, bot_name, embed_req_q, embed_res_q)
        for (s_idx, i_idx), vec in zip(insight_coordinates, i_vectors):
            new_summaries[s_idx]['insights'][i_idx]['embedding'] = vec

    # Speicher den aktualisierten Zustand ab
    unified_memory['cluster_summaries'] = new_summaries

    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    unified_file = f"{bot_base_dir}/database/unified_memory.pkl"

    # Der bestehende Speicher-Code folgt hier nativ:
    with open(unified_file, 'wb') as f:
        pickle.dump(unified_memory, f)

    # ============ ÄNDERUNG 9: Lösche geclusterte Fakten ============
    clustered_fact_indices = set([idx for idx in range(num_facts)
                                  if any(idx in cluster['fact_indices'] and idx < fact_offset
                                         for cluster in clusters_info)])

    new_facts = []
    new_metadata = {}
    for old_idx in range(len(unified_memory['facts'])):
        if old_idx not in clustered_fact_indices:
            new_idx = len(new_facts)
            new_facts.append(unified_memory['facts'][old_idx])
            if old_idx in unified_memory['metadata']:
                new_metadata[new_idx] = unified_memory['metadata'][old_idx]

    # ============ ÄNDERUNG 10: Aktualisiere unified_memory ============
    deleted_facts = len(unified_memory['facts']) - len(new_facts)
    unified_memory['facts'] = new_facts
    unified_memory['metadata'] = new_metadata
    unified_memory['cluster_summaries'] = new_summaries

    if 'cluster_history' not in unified_memory:
        unified_memory['cluster_history'] = []

    unified_memory['cluster_history'].append({
        'timestamp': pd.Timestamp.now().isoformat(),
        'num_clusters': target_clusters,
        'total_items_clustered': num_items,
        'new_summaries': len(new_summaries_to_create),
        'kept_summaries': len(old_summaries_to_keep),
        'facts_deleted': deleted_facts,
        'compression_factor': args.compression_factor
    })

    # ============ ÄNDERUNG 11: Speichere ============
    with open(unified_file, 'wb') as f:
        pickle.dump(unified_memory, f)

    print(f"{bot_name}: 🗑️  Deleted {deleted_facts} facts")
    print(f"{bot_name}: ✅ Created {len(new_summaries_to_create)} new summaries, kept {len(old_summaries_to_keep)} old summaries")
    print(f"{bot_name}: ✅ Updated unified consolidation: {len(new_facts)} facts + {len(new_summaries)} summaries remaining")


    # ============ Analysis before consolidation
    try:
        from consolidation.analysis import plot_memory_snapshot
        plot_memory_snapshot(bot_name = bot_name,
                             state="after", labels=labels)
    except Exception as plot_err:
        print(f"⚠️  Konnte Vor-Consolidation-Snapshot nicht erstellen: {plot_err}")


    force_save(bot_name, logger)


def force_save(bot_name, logger):
    """Extrahiere die vollständigen LLM-Responses und speichere sie als Pickle"""

    reasoning_input = f"bots/{bot_name}/database/ours_reasoning.jsonl"
    os.makedirs(f"bots/{bot_name}/database", exist_ok=True)

    output_dir = f"bots/{bot_name}/database/"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(reasoning_input):
        print(f"❌ Fehler: {reasoning_input} nicht gefunden!")
        return

    reasoning_units = []
    total_processed = 0
    total_failed = 0

    with open(reasoning_input, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_processed += 1

                # Die response IST die wichtige Zusammenfassung!
                response_text = data.get('response', '')

                if not response_text:
                    total_failed += 1
                    continue

                # Parse die Response um extended_insight zu extrahieren
                resp = fix_incomplete_json(response_text)

                if resp and "extended_insight" in resp:
                    # Speichere die gesamte Response + Metadaten
                    reasoning_units.append({
                        "content": response_text,  # ← Die vollständige Response!
                        "cluster_id": data.get('cluster_id', 0),
                        "item_count": data.get('item_count', 1),
                        "insights_count": len(resp.get("extended_insight", [])),
                        "model": data.get('model', 'unknown'),
                        "usage_tokens": data.get('usage', {}).get('total_tokens', 0)
                    })
                else:
                    total_failed += 1
            except Exception as e:
                total_failed += 1
                print(f"⚠️  Fehler beim Verarbeiten einer Zeile: {e}")

    if not reasoning_units:
        print("❌ Keine gültigen Reasoning Units gefunden!")
        return

    # Erstelle DataFrame und speichere
    final_df = pd.DataFrame(reasoning_units)
    output_file = os.path.join(output_dir, "reasoning_units.pkl")

    with open(output_file, 'wb') as f:
        pickle.dump(final_df, f)

    logger.info(f"✅ ERFOLG: {len(reasoning_units)} Reasoning Units extrahiert!")
    logger.info(f"   Verarbeitet: {total_processed}, Fehlgeschlagen: {total_failed}")
    logger.info(f"   Gespeichert in: {output_file}")

    # ============ ÄNDERUNG 1: Zeige Unified Memory Status (mit neuer Struktur) ============
    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    unified_file = f"{bot_base_dir}/database/unified_memory.pkl"
    if os.path.exists(unified_file):
        with open(unified_file, 'rb') as f:
            unified_memory = pickle.load(f)
        print(f"{bot_name}: \n📊 Unified Memory Status:")
        print(f"{bot_name}:    - Verbleibende Fakten: {len(unified_memory.get('facts', []))}")
        print(f"{bot_name}:    - Gespeicherte Cluster-Summaries: {len(unified_memory.get('cluster_summaries', []))}")

        # ============ ÄNDERUNG 2: Zeige Cluster History ============
        cluster_history = unified_memory.get('cluster_history', [])
        if cluster_history:
            print(f"{bot_name}:    - Clustering Runs: {len(cluster_history)}")
            latest = cluster_history[-1]
            print(f"{bot_name}:    - Letzter Run: {latest['timestamp']}")
            print(f"{bot_name}:      • Cluster erstellt: {latest['num_clusters']}")
            print(f"{bot_name}:      • Items geclustert: {latest['total_items_clustered']}")
            # Neue oder alte Keys - je nachdem was existiert
            if 'new_summaries' in latest:
                print(f"{bot_name}:      • Neue Summaries: {latest['new_summaries']}")
                print(f"{bot_name}:      • Beibehaltene alte Summaries: {latest.get('kept_summaries', 0)}")
            elif 'summaries_created' in latest:
                print(f"{bot_name}:      • Cluster-Summaries erstellt: {latest['summaries_created']}")

    print_memory(bot_name)


def print_memory(bot_name):

    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    unified_file = f"{bot_base_dir}/database/unified_memory.pkl"

    with open(unified_file, 'rb') as f:
        unified_memory = pickle.load(f)

    print("\n" + "=" * 80)
    print(f"{bot_name}: CLUSTER-ZUSAMMENFASSUNGEN (RAG Memory)")
    print("=" * 80)

    # Alle Cluster-Summaries
    summaries = unified_memory['cluster_summaries']
    print(f"Total clusters: {len(summaries)}")

    for i, summary in enumerate(summaries):
        print(f"\n🔹 Cluster {i}:")
        print(f"   Items: {summary['item_count']}")
        print(f"   Created: {summary['timestamp']}")
        for insight in summary['insights']:
            print(f"   • [{insight['inference_type'], insight['date']}] {insight['key']}: {insight['value']}")

    print("\n")


def start_rag(query_text, observation, bot_name, embed_req_q, embed_res_q):

    bot_base_dir = f"bots/{bot_name}"
    unified_memory_path=f"{bot_base_dir}/database/unified_memory.pkl"
    k = 10

    """
    Sucht hocheffizient in den vorausberechneten Vektoren einzelner Insights.
    Verhindert das datenbankseitige Neuerstellen von Embeddings zur Abfragezeit.
    """
    import os
    import pickle
    import numpy as np

    if not os.path.exists(unified_memory_path):
        print(f"⚠️ {unified_memory_path} nicht gefunden! Nutzen Sie Fallback.")
        return []

    with open(unified_memory_path, 'rb') as f:
        unified_memory = pickle.load(f)

    cluster_summaries = unified_memory.get('cluster_summaries', [])

    flat_insights = []
    vectors = []

    # 1. Sammle alle flachen Insights, die bereits einen Vektor besitzen
    for summary in cluster_summaries:
        cluster_timestamp = summary.get('timestamp', 'unknown')
        for insight in summary.get('insights', []):
            if 'embedding' in insight and insight['embedding'] is not None:
                key = insight.get('key', '').strip()
                value = insight.get('value', '').strip()
                inf_type = insight.get('inference_type', insight.get('type', 'info')).strip()
                #insight_date = insight.get('date', cluster_timestamp)

                insight_date = format_relative_time(insight.get('date', 'N/A').strip(), observation)

                formatted_text = f"[{insight_date}] {key}: {value}"

                flat_insights.append({
                    'text': formatted_text,
                })
                vectors.append(insight['embedding'])

    if not vectors:
        print("⚠️ Keine vorausberechneten Insight-Vektoren in der Memory-Datei gefunden.")
        return []

    # 2. Generiere NUR das Query-Embedding (1 einzige API-Anfrage)
    #query_vector = db.embedder.create([query_text], show_progress_bar=False)[0]
    raw_query_vector = get_shared_embedding([query_text], bot_name, embed_req_q, embed_res_q)
    query_vector = np.array(raw_query_vector).flatten()

    # 3. Berechne Vektorisierte Kosinus-Ähnlichkeit blitzschnell via NumPy
    X_insights = np.array(vectors)

    # Normalisierungen für die exakte Cosine-Similarity
    X_norm = X_insights / np.linalg.norm(X_insights, axis=1, keepdims=True)
    q_norm = query_vector / np.linalg.norm(query_vector)

    # Matrix-Multiplikation liefert alle Scores gleichzeitig
    scores = X_norm @ q_norm

    # Top-K Indizes sortieren (absteigend)
    top_k_indices = np.argsort(scores)[::-1][:k]

    # 4. Ergebnisse kompilieren & für Prompt formatieren
    results_strings = []

    for idx in top_k_indices:
        score = float(scores[idx])

        # Eine sinnvolle Schwelle für die Kosinus-Ähnlichkeit (z.B. 0.15 oder 0.2)
        # verhindert, dass völlig unpassende Fakten den Kontext überladen
        if score > 0.15:
            entry = flat_insights[idx]
            text_content = entry['text']

            # Formatiere jeden Fakt als sauberen Listenpunkt inklusive zeitlichem Kontext
            results_strings.append(f"- {text_content}")

    # Verbinde die Listenpunkte mit Zeilenumbrüchen.
    # Falls keine Treffer über der Schwelle lagen, geben wir einen leeren Fallback an.
    feedback_block = "\n".join(
        results_strings) if results_strings else "- No relevant insights available for this context."

    # Jetzt kannst du den fertigen String sauber injizieren
    enhanced_prompt = (
        f"### Abstract ideas as feedback:\n"
        f"{feedback_block}\n"
    )

    return enhanced_prompt


def format_relative_time(date_str, observation):
    """
    Converts a timestamp or time range into a relative natural language description.
    Handles formats like:
    - "2026-01-07 Wednesday 16:58:43"
    - "2026-01-07 Wednesday 15:13:49 to 2026-01-07 Wednesday 16:49:41"
    """
    try:
        # Reference time for calculation
        now = observation.get("time", "")

        raw_now = observation.get("time", "").replace("   - Current Time: ", "").strip()

        def parse_dt(s):
            s = s.strip()
            formats = [
                "%Y-%m-%d %A %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %A %H:%M",
                "%Y-%m-%d"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            return datetime.fromisoformat(s.split(' ')[0])

        now = parse_dt(raw_now)

        duration_str = ""
        if " to " in date_str:
            start_str, end_str = date_str.split(" to ")
            dt_start = parse_dt(start_str)
            dt_end = parse_dt(end_str)

            # Calculate duration
            duration = dt_end - dt_start
            d_hours = duration.seconds // 3600
            d_mins = (duration.seconds % 3600) // 60

            if duration.days > 0:
                duration_str = f" for {duration.days} days"
            elif d_hours > 0:
                duration_str = f" for {d_hours} hours"
            elif d_mins > 0:
                duration_str = f" for {d_mins} minutes"

            dt = dt_start  # Use start time for relative calculation
        else:
            dt = parse_dt(date_str)

        diff = now - dt
        days = diff.days
        seconds = diff.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60

        time_str = dt.strftime("%H:%M:%S")

        if days == 0:
            if hours == 0:
                if minutes == 0:
                    return f"Just now{duration_str} at {time_str}"
                return f"{minutes} minutes ago{duration_str} at {time_str}"
            return f"Today, {hours} hours ago{duration_str} at {time_str}"
        elif days == 1:
            return f"Yesterday{duration_str} at {time_str}"
        elif days < 7:
            return f"{days} days ago{duration_str} at {time_str}"
        else:
            weeks = days // 7
            return f"{weeks} weeks ago{duration_str} at {time_str}"

    except Exception:
        return date_str


def get_shared_embedding(text, bot_name, req_queue, res_queue):
    """
    Fragt den zentralen GPU-Worker nach dem Embedding für einen Text.
    """
    # Anfrage abschicken
    req_queue.put((bot_name, text))

    # Auf Antwort warten (blockiert nur diesen einen Bot-Prozess, nicht die GPU)
    vector = res_queue.get()
    return vector
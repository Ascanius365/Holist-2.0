import sys
import os
import json
import hashlib
import re
import ast
import warnings
import logging
import requests
import pandas as pd
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
from collections import defaultdict, deque
from neo4j import GraphDatabase

from neo4j import GraphDatabase
import numpy as np
from simple_chalk import chalk

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


# =====================================================================
# OPENROUTER RERANKER HELPER
# =====================================================================
def openrouter_rerank(query: str, documents: list, top_n: int = 4):
    """
    Nutzt den OpenRouter Rerank Endpunkt, um Dokumente semantisch bezüglich einer Query zu sortieren.
    """

    model = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

    # Statt hart codiertem Key:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY ist nicht in den Umgebungsvariablen gesetzt!")

    url = "https://openrouter.ai/api/v1/rerank"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Holist Minecraft Agent"
    }

    formatted_docs = [{"text": doc} if isinstance(doc, str) else doc for doc in documents]

    payload = {
        "model": model,
        "query": query,
        "documents": formatted_docs,
        "top_n": min(top_n, len(documents))
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            print(f"❌ OpenRouter Rerank Fehler: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Netzwerkfehler beim Reranking: {e}")
        return []


# =====================================================================
# JSON PARSING & UTILS
# =====================================================================
def fix_incomplete_json(json_str, session_id="Unbekannt"):
    if isinstance(json_str, dict):
        return json_str
    if not isinstance(json_str, str):
        return {}

    if "<think>" in json_str:
        json_str = json_str.split("</think>")[-1].strip()
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    fixed_str = json_str
    in_string = False
    escape = False
    stack = []

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

    if in_string:
        fixed_str += '"'

    while stack:
        open_char = stack.pop()
        if open_char == '{':
            fixed_str += '}'
        elif open_char == '[':
            fixed_str += ']'

    try:
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        pass

    try:
        cleaned_str = re.sub(r',\s*([\]}])', r'\1', fixed_str)
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        pass

    try:
        evaluated = ast.literal_eval(json_str)
        if isinstance(evaluated, dict):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    print(f"⚠️ JSON für Session '{session_id}' nicht reparierbar!")
    return {}


def get_fact_hash(fact_dict):
    insights_str = json.dumps(fact_dict.get('extended_insight', []), sort_keys=True)
    fact_str = json.dumps({
        'session_id': fact_dict.get('session_id'),
        'date': fact_dict.get('date'),
        'extended_insight': insights_str
    }, sort_keys=True)
    return hashlib.md5(fact_str.encode()).hexdigest()


def run_memory_pipeline(args, bot_name, query_text, observation, embed_req_q, embed_res_q, logger):
    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    unified_file = f"{bot_base_dir}/database/unified_memory.json"
    input_file = f"bots/{bot_name}/memory/sessions.jsonl"

    if os.path.exists(unified_file):
        with open(unified_file, 'r', encoding='utf-8') as f:
            unified_memory = json.load(f)
    else:
        unified_memory = {
            'facts': [],
            'processed_hashes': [],
            'similarity_matrix': {},
            'cluster_summaries': [],
            'version': 1
        }

    if 'similarity_matrix' not in unified_memory:
        unified_memory['similarity_matrix'] = {}

    active_lines = []
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        active_lines = [l for l in lines if l.strip()]

    dirty_clusters = [c for c in unified_memory.get('cluster_summaries', []) if c.get('needs_consolidation', False)]

    # Hier sammeln wir sowohl neue Fakten als auch aktualisierte Cluster-Summaries
    new_memory_entries = []

    if active_lines or dirty_clusters:
        logger.info("⚡ Starte asynchrone parallele Batch-Verarbeitung über ThreadPool...")
        tasks = []

        with ThreadPoolExecutor() as executor:
            # TASK A: Neue Session-Zeilen absenden
            if active_lines:
                for line_idx, line in enumerate(active_lines):
                    try:
                        data = json.loads(line)
                        inf_type = data.get("information_type") or data.get("type") or "observation"
                        date_str = data.get("session_date") or data.get("date") or datetime.now().strftime(
                            "%Y-%m-%d %A %H:%M:%S")
                        text_str = data.get("text") or data.get("response") or ""
                        if isinstance(text_str, dict):
                            text_str = json.dumps(text_str)
                        single_fragment_text = f"[{inf_type.lower()}, {date_str}]: {text_str}\n"
                    except json.JSONDecodeError:
                        single_fragment_text = f"[observation, {datetime.now().strftime('%Y-%m-%d')}]: {line.strip()}\n"

                    future_frag = executor.submit(run_llm_reasoning_info, single_fragment_text, bot_name)
                    tasks.append({"type": "fragments", "future": future_frag, "text": single_fragment_text,
                                  "line_idx": line_idx})

            # TASK B: Gedriftete Cluster zur Neukonsolidierung absenden
            if dirty_clusters:
                fact_map = {get_fact_hash(f): f for f in unified_memory.get('facts', [])}
                for cluster in dirty_clusters:
                    c_id = cluster.get("cluster_id")
                    hashes = cluster.get("associated_hashes", [])
                    cluster_facts = [fact_map[h] for h in hashes if h in fact_map]

                    if not cluster_facts:
                        cluster["needs_consolidation"] = False
                        continue

                    cluster_text = ""
                    for f in cluster_facts:
                        for insight in f.get('extended_insight', []):
                            inf_type = insight.get('inference_type', 'Fact')
                            date_str = insight.get('date', f.get('date', datetime.now().strftime("%Y-%m-%d")))
                            cluster_text += f"[{inf_type.lower()}, {date_str}]: {insight.get('key')}: {insight.get('value')}\n"

                    future_cluster = executor.submit(run_llm_reasoning_info, cluster_text, bot_name)
                    tasks.append({"type": "cluster", "cluster_obj": cluster, "future": future_cluster})

            # TASK C: Ergebnisse einsammeln
            processed_hashes_set = set(unified_memory.get('processed_hashes', []))
            for task in tasks:
                try:
                    llm_res = task["future"].result()
                    if not llm_res or not isinstance(llm_res, dict):
                        continue

                    # Trenne Insights und Query sauber auf
                    insights = llm_res.get("extended_insight", [])
                    query_text = llm_res.get("query", "")

                    if task["type"] == "fragments":
                        print("Generated a fact")
                        fact_dict = {
                            "session_id": f"reasoned_{datetime.now().strftime('%Y%m%d_%H%M%S')}_line{task['line_idx']}",
                            "text": task["text"],
                            "date": insights[0].get("date", datetime.now().strftime(
                                "%Y-%m-%d")) if insights else datetime.now().strftime("%Y-%m-%d"),
                            "query": query_text,  # 🚀 NEU: Query wird im Fakt gespeichert
                            "extended_insight": insights,
                            "type": "fact"  # Identifikator
                        }
                        if get_fact_hash(fact_dict) not in processed_hashes_set:
                            new_memory_entries.append(fact_dict)

                    elif task["type"] == "cluster":
                        print("Generated a Cluster")
                        cluster = task["cluster_obj"]
                        cluster["extended_insight"] = insights
                        cluster["query"] = query_text  # 🚀 NEU
                        cluster["needs_consolidation"] = False
                        c_id = cluster.get("cluster_id")

                        # WICHTIG: Alte Version der Summary dieses Clusters aus den lokalen Listen löschen
                        unified_memory['facts'] = [f for f in unified_memory.get('facts', []) if not (
                                f.get('type') == 'cluster_summary' and f.get('cluster_id') == c_id)]

                        cluster_fact = {
                            "session_id": f"cluster_summary_{c_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "query": query_text,  # 🚀 NEU
                            "extended_insight": insights,
                            "type": "cluster_summary",  # Identifikator
                            "cluster_id": c_id
                        }
                        new_memory_entries.append(cluster_fact)
                        print(f"✅ Parallel-Ergebnis: Cluster {c_id} erfolgreich als Summary-Fakt konsolidiert.")

                except Exception as thread_err:
                    logger.error(f"❌ Fehler bei der parallelen Thread-Ausführung: {thread_err}")

    # =====================================================================
    # 3. UNIFIZIERTES MATRIX-UPDATE FÜR FAKTEN & CLUSTER-SUMMARIES
    # =====================================================================
    if new_memory_entries:
        logger.info(
            f"📊 Matrix-Update: Berechne Ähnlichkeiten für {len(new_memory_entries)} neue Einträge (Fakten & Summaries)...")

        all_top_matches = []
        all_deleted_hashes = set()

        for new_entry in new_memory_entries:
            # Erzeugt eine saubere, ausformulierte Beschreibung aus den Insights für den Reranker
            if 'extended_insight' in new_entry and not new_entry.get('description'):
                new_entry['description'] = " ".join([i.get('value', '') for i in new_entry['extended_insight']])
            existing_facts = unified_memory.get('facts', [])
            pool = []
            new_hash = get_fact_hash(new_entry)
            new_query = new_entry.get('query', '')

            if existing_facts:
                for f in existing_facts:
                    text_content = f.get('description', '')
                    pool.append(f"[{f.get('date', 'N/A')}] ({f.get('type', 'fact')}) {text_content}")

            rerank_results = []
            if pool:
                print(f"🔎 Reranking neuen {new_entry['type']} gegen {len(pool)} bestehende Einträge...")
                rerank_results = openrouter_rerank(query=new_query, documents=pool, top_n=len(pool))

            if new_hash not in unified_memory['similarity_matrix']:
                unified_memory['similarity_matrix'][new_hash] = {}

            # Set zum Sammeln von redundanten Altfakten (Score > 0.9)
            hashes_to_delete = set()

            for res in rerank_results:
                rel_idx = res['index']
                score = res['relevance_score']

                target_fact = existing_facts[rel_idx]
                target_hash = get_fact_hash(target_fact)

                # 🎯 Redundanz-Filter: Wenn die Ähnlichkeit > 0.9 ist, wird der alte Eintrag vorgemerkt
                if score > 0.95:
                    print(
                        f"🗑️ Redundanz erkannt: Fakt {target_hash} hat Score {score:.4f} > 0.95 und wird gelöscht.")
                    hashes_to_delete.add(target_hash)
                    all_deleted_hashes.add(target_hash)
                    continue  # Keine Kante für diesen gelöschten Fakt erstellen

                # Rein gerichtete Verknüpfung (Vom neuen Fakt zum alten)
                unified_memory['similarity_matrix'][new_hash][target_hash] = score

                all_top_matches.append((score, target_fact))

            # 🛠️ Bereinigung des lokalen Speichers nach dem Reranking dieses Eintrags
            if hashes_to_delete:
                # 1. Aus der Liste der Fakten entfernen
                unified_memory['facts'] = [f for f in unified_memory['facts'] if
                                           get_fact_hash(f) not in hashes_to_delete]

                # 2. Aus den verarbeiteten Hashes filtern
                unified_memory['processed_hashes'] = [h for h in unified_memory['processed_hashes'] if
                                                      h not in hashes_to_delete]

                # 3. Aus der Ähnlichkeitsmatrix komplett austragen
                for h in hashes_to_delete:
                    unified_memory['similarity_matrix'].pop(h, None)
                for source_hash in unified_memory['similarity_matrix']:
                    for h in hashes_to_delete:
                        unified_memory['similarity_matrix'][source_hash].pop(h, None)

            # Jetzt erst den neuen Eintrag lokal anhängen
            unified_memory['facts'].append(new_entry)
            if new_hash not in unified_memory['processed_hashes']:
                unified_memory['processed_hashes'].append(new_hash)

        # Speicher sichern
        with open(unified_file, 'w', encoding='utf-8') as f:
            json.dump(unified_memory, f, indent=2, ensure_ascii=False)

        db_name = bot_name.lower()

        try:
            unified_memory = sync_memory_to_neo4j(
                unified_memory,
                database=db_name,  # 🎯 Hier wird die bot-spezifische DB übergeben
                uri="bolt://localhost:7687",
                auth=("neo4j", "eher2015")
            )
            logger.info(f"💾 Alle Daten erfolgreich mit Neo4j (Datenbank: {db_name}) abgeglichen.")
        except Exception as neo_err:
            logger.error(f"❌ Neo4j-Pipeline für Datenbank '{db_name}' fehlgeschlagen: {neo_err}")

        # Speicher sichern
        with open(unified_file, 'w', encoding='utf-8') as f:
            json.dump(unified_memory, f, indent=2, ensure_ascii=False)

        if os.path.exists(input_file) and active_lines:
            with open(input_file, 'w', encoding='utf-8') as f:
                f.writelines([])  # Verarbeitete Zeilen leeren

        # =====================================================================
        # 4. POWER GRID RETRIEVAL & HIERARCHISCHE AUSWERTUNG
        # =====================================================================
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "eher2015"))

        # Schritt A: Das globale, kreisfreie Stromnetzwerk berechnen
        grid_trajectories = precompute_cluster_power_grid(driver, db_name, limit=50)

        # Schritt B: Reranker-Ergebnisse sortieren und Einspeisepunkte (Kraftwerke) isolieren
        all_top_matches.sort(key=lambda x: x[0], reverse=True)
        seen_hashes = set()
        top_clusters = []  # Speichert Tupel aus (Score, Cluster-Fakt-Objekt)
        unique_top_facts = []

        for score, fact in all_top_matches:
            f_hash = get_fact_hash(fact)
            if f_hash in seen_hashes or f_hash in all_deleted_hashes:
                continue
            seen_hashes.add(f_hash)

            if fact.get('type') == 'cluster_summary' and score > 0.5:
                top_clusters.append((score, fact))

            if fact.get('type') == 'fact' and len(unique_top_facts) < 2:
                unique_top_facts.append((score, fact))

        # ---------------------------------------------------------------------
        # NEUER ANSATZ: Aufbau eines ATOMAREN Graphen aus den Pfadsegmenten
        # ---------------------------------------------------------------------
        from collections import defaultdict, deque

        atomic_graph = defaultdict(list)
        node_lookup = {}  # Speichert die echten Knoten-Objekte anhand ihres Hashes

        for edge in grid_trajectories:
            path_nodes = list(edge["path"].nodes)

            # Wir registrieren alle Knoten im Lookup-Table
            for node in path_nodes:
                n_hash = get_fact_hash(node)
                node_lookup[n_hash] = node

            # Wir zerlegen den Pfad in seine atomaren 1-Hop-Segmente
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                u_hash = get_fact_hash(u)
                v_hash = get_fact_hash(v)

                # Bidirektionales Hinzufügen der echten Nachbarschaft
                if v_hash not in atomic_graph[u_hash]:
                    atomic_graph[u_hash].append(v_hash)
                if u_hash not in atomic_graph[v_hash]:
                    atomic_graph[v_hash].append(u_hash)

        # Schritt C: Hierarchisches Ausschwärmen auf dem ATOMAREN Graphen (Radius 1-2)
        filtered_clusters = [c for c in top_clusters if c[0] >= 0.5]
        if not filtered_clusters and top_clusters:
            top_clusters.sort(key=lambda x: x[0], reverse=True)
            filtered_clusters = [top_clusters[0]]

        trajectories_feedback_lines = []

        for c_score, c_fact in filtered_clusters:
            c_hash = get_fact_hash(c_fact)
            c_desc = c_fact.get('description', 'Keine Beschreibung verfügbar')

            cluster_block = f"\n🔹 [Cluster-Summary | Reranker-Score: {c_score:.4f}]: {c_desc}\n"
            local_unique_nodes = {}

            # Fallback: Falls der exakte Hash nicht im Graphen ist, suchen wir nach einer ID- oder Textübereinstimmung
            start_hash = None
            if c_hash in atomic_graph:
                start_hash = c_hash
            else:
                # Tolerantes Matching über die Beschreibungen im Lookup
                for node_hash, node_obj in node_lookup.items():
                    if node_obj.get("type") == "cluster_summary" and node_obj.get("description") == c_desc:
                        start_hash = node_hash
                        break

            if start_hash and start_hash in atomic_graph:
                # Queue speichert: (aktueller_knoten_hash, atomare_distanz_zum_cluster)
                queue = deque([(start_hash, 0)])
                local_visited = {start_hash}

                while queue:
                    current_hash, current_dist = queue.popleft()

                    # Bei Distanz 2 sammeln wir die Nachbarn (welche dann Distanz 3 wären) nicht mehr
                    if current_dist >= 2:
                        continue

                    for neighbor_hash in atomic_graph[current_hash]:
                        if neighbor_hash not in local_visited:
                            local_visited.add(neighbor_hash)

                            neighbor_node = node_lookup.get(neighbor_hash)
                            if neighbor_node:
                                next_dist = current_dist + 1

                                # Nur echte Fakten sammeln
                                if neighbor_node.get("type") == "fact":
                                    local_unique_nodes[
                                        neighbor_hash] = f"[Distanz {next_dist}] {neighbor_node.get('description', '')}"

                                queue.append((neighbor_hash, next_dist))

            # Ausgabe der gefilterten Fakten für dieses Cluster
            if local_unique_nodes:
                # Sortierung nach Distanz für eine schönere UI-Hierarchie
                sorted_nodes = sorted(local_unique_nodes.values(), key=lambda x: x.startswith("[Distanz 2]"))
                node_lines = [f"  ├── 📌 {desc}" for desc in sorted_nodes]
                cluster_block += "\n".join(node_lines)
            else:
                cluster_block += "  └── (Keine assoziierten Fakten im atomaren Radius 1-2 gefunden)"

            trajectories_feedback_lines.append(cluster_block)

        driver.close()

        # Schritt D: Einzel-Fakten für das Context Window aufbereiten
        results_strings = []
        for score, f in unique_top_facts:
            results_strings.append(f"- [Score: {score:.4f}]: {f.get('description', '')}")

        # Finalen Prompt-Kontext sauber zusammensetzen
        feedback_block = "### 🛤️ Aktivierte Wissens-Cluster und deren Assoziationsketten:\n"
        if trajectories_feedback_lines:
            feedback_block += "\n".join(trajectories_feedback_lines)
        else:
            feedback_block += "- Keine Cluster-Summaries mit einem Score größer als 0.7 gefunden."

        feedback_block += "\n\n### 📌 Top Einzel-Fakten:\n"
        feedback_block += "\n".join(results_strings) if results_strings else "- Keine passenden Einzel-Fakten."

    else:
        feedback_block = "- No new entries processed."

    print(chalk.blue(f"Feedback block: {feedback_block}"))

    return f"### Abstract ideas as feedback:\n{feedback_block}\n"

# =====================================================================
# AUXILIARY UTILS
# =====================================================================
def format_relative_time(date_str, observation):
    try:
        raw_now = observation.get("time", "").replace("   - Current Time: ", "").strip()

        def parse_dt(s):
            s = s.strip()
            formats = ["%Y-%m-%d %A %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %A %H:%M", "%Y-%m-%d"]
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
            duration = dt_end - dt_start
            d_hours = duration.seconds // 3600
            d_mins = (duration.seconds % 3600) // 60
            if duration.days > 0:
                duration_str = f" for {duration.days} days"
            elif d_hours > 0:
                duration_str = f" for {d_hours} hours"
            elif d_mins > 0:
                duration_str = f" for {d_mins} minutes"
            dt = dt_start
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
                if minutes == 0: return f"Just now{duration_str} at {time_str}"
                return f"{minutes} minutes ago{duration_str} at {time_str}"
            return f"Today, {hours} hours ago{duration_str} at {time_str}"
        elif days == 1:
            return f"Yesterday{duration_str} at {time_str}"
        elif days < 7:
            return f"{days} days ago{duration_str} at {time_str}"
        else:
            return f"{days // 7} weeks ago{duration_str} at {time_str}"
    except Exception:
        return date_str


def print_memory_json(bot_name):
    unified_file = f"bots/{bot_name}/database/unified_memory.json"
    if not os.path.exists(unified_file):
        return
    with open(unified_file, 'r', encoding='utf-8') as f:
        unified_memory = json.load(f)

    print("\n" + "=" * 80)
    print(f"{bot_name}: CLUSTER-ZUSAMMENFASSUNGEN (JSON RAG Memory)")
    print("=" * 80)
    summaries = unified_memory.get('cluster_summaries', [])
    for i, summary in enumerate(summaries):
        print(f"\n🔹 Cluster {i}: {summary.get('description')}")
        for insight in summary.get('insights', []):
            print(f"   • [{insight.get('inference_type')}] {insight.get('key')}: {insight.get('value')}")
    print("\n")


import yaml


# =====================================================================
# NATIVE REASON_INFO.YAML COMPLETION HELPER
# =====================================================================
def run_llm_reasoning_info(fragments_text: str, bot_name: str, yaml_path: str = "consolidation/reason_info.yaml") -> list:
    """
    Lädt die Prompt-Konfiguration aus der YAML-Datei, ersetzt das Fragment-Placeholder
    und holt die strukturierten Insights über OpenRouter ein.
    """
    if not os.path.exists(yaml_path):
        print(f"❌ Prompt-Datei nicht gefunden unter: {yaml_path}")
        return []

    # 1. YAML-Datei dynamisch einlesen
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Fehler beim Laden der YAML-Datei: {e}")
        return []

    # 2. Prompt-Inhalt extrahieren und Platzhalter ersetzen
    base_prompt = config["messages"][0]["content"]
    full_prompt = base_prompt.replace("{{$memory_fragments}}", fragments_text)

    # OpenRouter API Setup
    url = "https://openrouter.ai/api/v1/chat/completions"
    # Statt hart codiertem Key:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY ist nicht in den Umgebungsvariablen gesetzt!")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": f"{bot_name} Reason-Info Ingestion"
    }

    # Payload dynamisch aus den YAML-Parametern speisen
    payload = {
        "model": "openai/gpt-4o-mini",  # Oder dein bevorzugtes Modell
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 1024)
    }

    try:
        # Timeout ebenfalls dynamisch aus der YAML ziehen
        response = requests.post(url, json=payload, headers=headers, timeout=config.get("timeout", 45))
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Nutze deine bestehende Funktion zur JSON-Reparatur
            parsed_json = fix_incomplete_json(content)
            return parsed_json
        else:
            print(f"❌ OpenRouter Ingestion Fehler ({response.status_code}): {response.text}")
            return []
    except Exception as e:
        print(f"❌ Netzwerkfehler beim Fact-Reasoning: {e}")
        return []


def sync_memory_to_neo4j(unified_memory, database, uri="bolt://localhost:7687", auth=("neo4j", "eher2015")):
    driver = GraphDatabase.driver(uri, auth=auth)
    cluster_mapping = {}

    facts_payload = []
    current_hashes = []

    # Da Summaries nun auch in 'facts' liegen, verarbeiten wir alles in einem Rutsch
    for fact in unified_memory.get('facts', []):
        f_hash = get_fact_hash(fact)
        current_hashes.append(f_hash)

        facts_payload.append({
            "hash": f_hash,
            "session_id": fact.get("session_id", ""),
            "date": fact.get("date", ""),
            "query": fact.get("query", ""),
            "description": fact.get("description", ""),
            "summary": fact.get("description", ""),  # Für Abwärtskompatibilität in Explore
            "type": fact.get("type", "fact"),
            "cluster_id": fact.get("cluster_id", -1),
            # 🚀 FIX 1: extended_insight_json muss als String in die Payload, da Neo4j es im SET aufruft!
            "extended_insight_json": json.dumps(fact.get("extended_insight", []), ensure_ascii=False)
        })

    edges_payload = []
    matrix = unified_memory.get('similarity_matrix', {})
    for source_hash, targets in matrix.items():
        for target_hash, score in targets.items():
            if source_hash == target_hash:
                continue
            if source_hash in current_hashes and target_hash in current_hashes:
                if score >= 0.05:
                    edges_payload.append({
                        "source": source_hash,
                        "target": target_hash,
                        "score": float(score)
                    })

    try:
        with driver.session(database=database) as session:
            session.run("CREATE CONSTRAINT fact_hash_idx IF NOT EXISTS FOR (f:Fact) REQUIRE f.hash IS UNIQUE")

            # Pruning: Löscht veraltete Fakten und veraltete Cluster-Summaries (da deren Hashes sich geändert haben)
            cleanup_nodes_query = "MATCH (f:Fact) WHERE NOT f.hash IN $current_hashes DETACH DELETE f"
            session.run(cleanup_nodes_query, current_hashes=current_hashes)
            session.run("MATCH (:Fact)-[r:SIMILAR_TO]->(:Fact) DELETE r")

            # Nodes einheitlich schreiben
            if facts_payload:
                # 🚀 FIX 2: f.query und f.description im SET hinzugefügt!
                fact_query = """
                            UNWIND $facts AS fact_data
                            MERGE (f:Fact {hash: fact_data.hash})
                            SET f.session_id = fact_data.session_id,
                                f.date = fact_data.date,
                                f.query = fact_data.query,
                                f.description = fact_data.description,
                                f.extended_insight_json = fact_data.extended_insight_json,
                                f.summary = fact_data.summary,
                                f.type = fact_data.type,
                                f.cluster_id = fact_data.cluster_id
                            """
                session.run(fact_query, facts=facts_payload)

            # Kanten schreiben (Verknüpft gleichermaßen fact->fact, fact->summary und summary->summary)
            if edges_payload:
                edge_query = """
                UNWIND $edges AS edge_data
                MATCH (source:Fact {hash: edge_data.source})
                MATCH (target:Fact {hash: edge_data.target})
                MERGE (source)-[r:SIMILAR_TO]->(target)
                SET r.score = edge_data.score
                """
                session.run(edge_query, edges=edges_payload)

                print("🧠 Führe Louvain-Clustering nativ in Neo4j aus...")
                session.run("CALL gds.graph.drop('memoryGraph', false)")
                # Das GDS-Projekt lädt nun alle :Fact Knoten (Fakten + Summaries) und deren SIMILAR_TO Beziehungen!
                session.run(
                    "CALL gds.graph.project('memoryGraph', 'Fact', { SIMILAR_TO: { orientation: 'UNDIRECTED', properties: 'score' } })"
                )
                session.run(
                    "CALL gds.louvain.write('memoryGraph', { writeProperty: 'cluster_id', relationshipWeightProperty: 'score', maxLevels: 1 })")
                session.run(
                    "CALL gds.fastRP.write('memoryGraph', { relationshipWeightProperty: 'score', embeddingDimension: 128, iterationWeights: [0.0, 1.0, 0.7], writeProperty: 'structure_embedding' })")
                session.run("CALL gds.graph.drop('memoryGraph')")

                print("📥 Synchronisiere neue topologische Cluster-Zuordnungen zurück...")
                result = session.run("MATCH (f:Fact) RETURN f.hash AS hash, f.cluster_id AS cluster_id")
                cluster_mapping = {record["hash"]: record["cluster_id"] for record in result}

            # Topologische Shifts der Cluster bestimmen
            unified_memory = compute_topological_cluster_embeddings(session, unified_memory)

    finally:
        driver.close()

    # Lokale IDs für konsistente Rückgabe spiegeln
    for fact in unified_memory.get('facts', []):
        f_hash = get_fact_hash(fact)
        fact['cluster_id'] = cluster_mapping.get(f_hash, -1)

    return unified_memory


def compute_topological_cluster_embeddings(session, unified_memory, shift_threshold=2.2):
    """
    Fragt FastRP-Embeddings ab, berechnet den topologischen Zentroiden pro Cluster
    und vergleicht ihn mit dem Vorzustand. Nutzt den Overlap von Fakten-Hashes für ein
    stabiles Tracking über dynamische Louvain-Re-Runs hinweg.
    """
    # 1. Struktur-Embeddings und Cluster-IDs aus Neo4j abfragen
    result = session.run("""
        MATCH (f:Fact) 
        WHERE f.structure_embedding IS NOT NULL AND f.cluster_id IS NOT NULL
        RETURN f.hash AS hash, f.cluster_id AS cluster_id, f.structure_embedding AS embedding
    """)

    # Daten nach Cluster gruppieren und Hashes sammeln
    cluster_vectors = {}
    cluster_hashes = {}
    for record in result:
        c_id = record['cluster_id']
        emb = record['embedding']  # Liste von Floats aus Neo4j (FastRP-Vektor)
        f_hash = record['hash']

        if c_id not in cluster_vectors:
            cluster_vectors[c_id] = []
            cluster_hashes[c_id] = []
        cluster_vectors[c_id].append(emb)
        cluster_hashes[c_id].append(f_hash)

    # Lokale alte Summaries als Pool für das Hash-Matching holen
    old_summaries = unified_memory.get('cluster_summaries', [])
    new_summaries = []

    # 2. Jedes von Louvain gefundene Cluster analysieren
    for c_id, vectors in cluster_vectors.items():
        current_hashes = cluster_hashes[c_id]
        current_hashes_set = set(current_hashes)

        # Mathematischer Mittelwert (Zentroid) über alle FastRP-Vektoren des Clusters
        vectors_np = np.array(vectors)
        new_centroid = np.mean(vectors_np, axis=0).tolist()

        # 🎯 STABILES MATCHING: Finde das alte Cluster mit dem größten Fakten-Overlap
        best_old_summary = None
        max_overlap = 0

        for old_s in old_summaries:
            old_hashes_set = set(old_s.get('associated_hashes', []))
            overlap = len(current_hashes_set.intersection(old_hashes_set))
            if overlap > max_overlap:
                max_overlap = overlap
                best_old_summary = old_s

        # Wenn ein valides Match existiert (mindestens 1 überlappender Fakt)
        if best_old_summary and max_overlap > 0:
            # 🔄 CLUSTER EXISTIERT BEREITS -> Auf topologischen Shift prüfen
            old_centroid = best_old_summary.get("topological_embedding")

            if old_centroid and len(old_centroid) == len(new_centroid):
                # Euklidische Distanz zwischen altem und neuem Schwerpunkt berechnen
                distance = float(np.linalg.norm(np.array(old_centroid) - np.array(new_centroid)))

                if distance > shift_threshold:
                    print(
                        f"🔄 Cluster {c_id} (ehemals {best_old_summary.get('cluster_id')}) hat sich verschoben (Shift: {distance:.4f} > {shift_threshold}). Markiere für LLM.")
                    best_old_summary["needs_consolidation"] = True
                # HINWEIS: Falls das LLM es in diesem Durchlauf frisch auf False gesetzt hat,
                # bleibt es False, es sei denn, der topologische Shift war zu radikal.
            else:
                best_old_summary["needs_consolidation"] = True

            # Metadaten auf die neue Louvain-Realität aktualisieren
            best_old_summary["cluster_id"] = c_id
            best_old_summary["topological_embedding"] = new_centroid
            best_old_summary["associated_hashes"] = current_hashes
            best_old_summary["description"] = f"Neo4j Cluster {c_id} ({len(current_hashes)} Fakten)"

            # Synchronisation für print_memory_json (Spiegelung der Keys)
            if "extended_insight" in best_old_summary and not best_old_summary.get("insights"):
                best_old_summary["insights"] = best_old_summary["extended_insight"]

            new_summaries.append(best_old_summary)

        else:
            # ✨ KOMPLETT NEUES CLUSTER ENTSTANDEN -> Sofort vormerken
            print(f"✨ Neues Cluster {c_id} via FastRP/Louvain entdeckt. Markiere für LLM.")
            new_summary = {
                "cluster_id": c_id,
                "description": f"Neo4j Cluster {c_id} ({len(current_hashes)} Fakten)",
                "associated_hashes": current_hashes,
                "insights": [],
                "extended_insight": [],
                "topological_embedding": new_centroid,
                "needs_consolidation": True  # 🎯 Vormerken fürs LLM!
            }
            new_summaries.append(new_summary)

    # Speicher aktualisieren
    unified_memory['cluster_summaries'] = new_summaries
    return unified_memory


def precompute_cluster_power_grid(driver, database, limit=30):
    """
    PHASE 1: Berechnet das globale, kreis- und redundanzfreie Stromnetzwerk.
    Sichert über bidirektionale Spiegelung, dass absolut jedes Clusterknoten-Paar
    optimal verdrahtet wird und schwache Knoten nicht durch starke Verdrängung isoliert werden.
    """
    query = """
    // =====================================================================
    // SUB-PHASE 1: Alle globalen Pfade ungerichtet finden und bewerten
    // =====================================================================
    MATCH path = (c1)-[*2..4]-(c2)
    WHERE c1.type = "cluster_summary"
      AND c2.type = "cluster_summary"
      AND id(c1) < id(c2)
      AND ALL(n IN nodes(path)[1..-1] WHERE n.type = "fact")
      AND size([n IN nodes(path) | id(n)]) = size(reduce(s = [], n IN nodes(path) | CASE WHEN id(n) IN s THEN s ELSE s + id(n) END))

    WITH path, c1, c2,
         reduce(logScore = 0.0, r IN relationships(path) | logScore + log(r.score)) / size(relationships(path)) AS avg_log_score
    WITH path, c1, c2, exp(avg_log_score) AS path_score

    ORDER BY path_score DESC
    WITH c1, c2, collect(path)[0] AS bester_pfad, collect(path_score)[0] AS bester_score

    // =====================================================================
    // SUB-PHASE 2: Echte bidirektionale Rettungsanker für JEDES Cluster sichern
    // =====================================================================
    WITH collect({fokus: c1, c1_hash: c1.hash, c2_hash: c2.hash, pfad: bester_pfad, score: bester_score}) +
         collect({fokus: c2, c1_hash: c1.hash, c2_hash: c2.hash, pfad: bester_pfad, score: bester_score}) AS spiegel_liste

    UNWIND spiegel_liste AS sicht
    WITH sicht.fokus AS knoten, sicht
    ORDER BY sicht.score DESC

    WITH knoten, collect({c1_hash: sicht.c1_hash, c2_hash: sicht.c2_hash, pfad: sicht.pfad, score: sicht.score})[0] AS anker_pfad
    WITH collect(anker_pfad {.*, ist_anker: true}) AS markierte_anker

    // =====================================================================
    // SUB-PHASE 3: Alle Pfade einsammeln und mit den Ankern verschmelzen
    // =====================================================================
    MATCH path = (c1)-[*2..4]-(c2)
    WHERE c1.type = "cluster_summary"
      AND c2.type = "cluster_summary"
      AND id(c1) < id(c2)
      AND ALL(n IN nodes(path)[1..-1] WHERE n.type = "fact")
      AND size([n IN nodes(path) | id(n)]) = size(reduce(s = [], n IN nodes(path) | CASE WHEN id(n) IN s THEN s ELSE s + id(n) END))

    WITH path, c1, c2, markierte_anker,
         reduce(logScore = 0.0, r IN relationships(path) | logScore + log(r.score)) / size(relationships(path)) AS avg_log_score
    WITH path, c1, c2, exp(avg_log_score) AS path_score, markierte_anker

    ORDER BY path_score DESC
    WITH c1, c2, collect(path)[0] AS bester_pfad, collect(path_score)[0] AS bester_score, markierte_anker

    WITH {c1_hash: c1.hash, c2_hash: c2.hash, pfad: bester_pfad, score: bester_score, ist_anker: false} AS standard_pfad, markierte_anker
    WHERE NOT ANY(anker IN markierte_anker WHERE anker.pfad = standard_pfad.pfad)

    WITH markierte_anker, collect(standard_pfad) AS rest_pfade
    WITH markierte_anker + rest_pfade AS alle_eindeutigen_pfade

    UNWIND alle_eindeutigen_pfade AS element

    // =====================================================================
    // SUB-PHASE 4: Finale Sortierung (Anker werden zum Überleben gezwungen)
    // =====================================================================
    WITH element
    ORDER BY element.ist_anker DESC, element.score DESC
    LIMIT $limit

    RETURN element.pfad AS path, 
           element.c1_hash AS c1_hash, 
           element.c2_hash AS c2_hash, 
           element.score AS path_score
    """
    grid_trajectories = []
    try:
        with driver.session(database=database) as session:
            result = session.run(query, limit=limit)
            for record in result:
                grid_trajectories.append({
                    "path": record["path"],
                    "c1_hash": record["c1_hash"],
                    "c2_hash": record["c2_hash"],
                    "score": record["path_score"]
                })
    except Exception as e:
        print(f"⚠️ Fehler bei globaler Trajektorien-Extraktion: {e}")
    return grid_trajectories
import ast
import asyncio
import colorsys
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import umap
import yaml
from openai import AsyncOpenAI
from sklearn.cluster import HDBSCAN
from simple_chalk import chalk

from dotenv import load_dotenv

"""
env_path = Path(__file__).parent / '.env'
success = load_dotenv(dotenv_path=env_path)"""

# API-Konfiguration
OPENROUTER_KEY = ""

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

TEACHER_MODEL = "openai/gpt-4o-mini"
AGENT_MODEL = "openai/gpt-4o-mini"


async def run_memory_pipeline(bot_name, logger, memory, query):
    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}/database", exist_ok=True)
    input_file = f"bots/{bot_name}/memory/sessions.jsonl"

    # 1. Nur wirklich neue, unverarbeitete Zeilen aus der sessions.jsonl einlesen[cite: 2]
    active_lines = []

    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            # JETZT WERDEN DIE ZEILEN AUCH TATSÄCHLICH HINZUGEFÜGT:
            active_lines.append(cleaned_line)

    new_memory_entries = []

    if active_lines:
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

            for task in tasks:
                try:
                    llm_res = task["future"].result()
                    if not llm_res or not isinstance(llm_res, dict):
                        continue

                    insights_list = llm_res.get("extended_insight", [])
                    insight = insights_list[0] if isinstance(insights_list, list) and len(insights_list) > 0 else {}

                    fact_dict = {
                        "session_id": f"reasoned_{datetime.now().strftime('%Y%m%d_%H%M%S')}_line{task['line_idx']}",
                        "date": insight.get("date", datetime.now().strftime("%Y-%m-%d")),
                        "text": str(insight.get("value", "")),
                        "abstraction": float(insight.get("abstraction", 0.2))
                    }
                    new_memory_entries.append(fact_dict)

                except Exception as thread_err:
                    logger.error(f"❌ Fehler bei der parallelen Thread-Ausführung: {thread_err}")

    if os.path.exists(input_file) and active_lines:
        with open(input_file, 'w', encoding='utf-8') as f:
            f.writelines([])  # Verarbeitete Zeilen leeren

    print("🚀 Starte sequenziellen kognitiven Gedächtnis-Stream...")
    retrieved_cluster = ""
    if new_memory_entries:
        for new_entry in new_memory_entries:
            text = f"[{new_entry.get("date")}] {new_entry.get("text")}"
            abstraction_score = new_entry.get("abstraction")
            print(f"Text: {text} | Abstraction score: {abstraction_score}")

            retrieved_cluster = await memory.add_and_consolidate_entry(text, abstraction_score)

    # Speicher sichern
    memory.save(bot_name)

    # Abschließendes Clustering plotten
    visualize_final_memory(bot_name, memory)

    print(f"Retrieved cluster: {retrieved_cluster}")

    return retrieved_cluster


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


def run_llm_reasoning_info(fragments_text: str, bot_name: str, yaml_path: str = "consolidation/reason_info2.yaml") -> dict:
    """
    Lädt die Prompt-Konfiguration aus der YAML-Datei, ersetzt das Fragment-Placeholder
    und holt die strukturierten Insights über OpenRouter ein.
    """
    if not os.path.exists(yaml_path):
        print(f"❌ Prompt-Datei nicht gefunden unter: {yaml_path}")
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Fehler beim Laden der YAML-Datei: {e}")
        return {}

    base_prompt = config["messages"][0]["content"]
    full_prompt = base_prompt.replace("{{$memory_fragments}}", fragments_text)

    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY ist nicht in den Umgebungsvariablen gesetzt!")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": f"{bot_name} Reason-Info Ingestion"
    }

    payload = {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 1024)
    }

    #print(chalk.blue(f"Prompt 1: {full_prompt}"))

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=config.get("timeout", 45))
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            return fix_incomplete_json(content)
        else:
            print(f"❌ OpenRouter Ingestion Fehler ({response.status_code}): {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Netzwerkfehler beim Fact-Reasoning: {e}")
        return {}


class HierarchicalVectorMemory:
    def __init__(self, bot_name, openrouter_api_key=OPENROUTER_KEY):
        base_path = "hierarchical_memory"
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Bitte setze den OpenRouter API-Key.")

        self.openrouter_url = "https://openrouter.ai/api/v1/embeddings"
        self.embedding_model = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

        self.embeddings = []  # Liste von np.array (2048D)
        self.hybrid_vectors = []  # Liste von np.array (2049D: Embedding + Abstraction)
        self.metadata = []  # Liste von Dicts mit {"id": int, "text": str, "abstraction": float}
        self.id_counter = 0  # Fortlaufende ID für präzise Referenzierung im Agenten

        # 🚀 VERSUCHE ZUERST ZU LADEN. WENN DATEIEN FEHLEN, STARTE PRE-FILL
        if self.load(bot_name):
            print("ℹ️ Bestehender Speicher geladen. Kaltstart übersprungen.")
        else:
            demo_data = [
                "The bot often harvests the grain in the field in the morning",
                "The bot has sorted the chest containing 10 stones and seeds",
                "The bot frequently checks the furnace and most recently took three iron bars out of it",
                "The bot sees a garden bed in the field",
                "The bot lives in an area with plenty of forest and meadow, and abundant resources",
                "The bot can see a forest nearby where wood can be felled",
                "The bot sees several trees on the hill",
                "The bot has mined 5 oak logs",
                "The bot heard a strange chicken noise in the far distance yesterday"
            ]

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Task einplanen, falls bereits in einem async Loop gearbeitet wird
                loop.create_task(self.prefill_memory_async(demo_data))
            else:
                asyncio.run(self.prefill_memory_async(demo_data))


    async def prefill_memory_async(self, data_list):
        """
        Befüllt das Gedächtnis schnell und parallel mit einer Liste von Einträgen.
        Umgeht den Consolidation-Agenten für einen schnellen Kaltstart.
        """
        print(f"🚀 Initialisiere Speicher: Verarbeite {len(data_list)} Einträge parallel...")

        # 1. Erstelle alle Tasks für parallele Ausführung
        tasks = []
        for text in data_list:
            score_task = self.calculate_api_abstraction_score2(text)
            loop = asyncio.get_running_loop()
            embedding_task = loop.run_in_executor(None, self._get_openrouter_embedding, text, "passage")
            # Gruppiere Score und Embedding für jeden einzelnen Text
            tasks.append(asyncio.gather(score_task, embedding_task))

        # 2. Alle API-Calls parallel abfeuern
        results = await asyncio.gather(*tasks)

        # 3. Ergebnisse sequenziell im Speicher ablegen
        for text, (abstraction_score, emb) in zip(data_list, results):
            hybrid_vec = np.append(emb, abstraction_score)
            entry_id = self.id_counter
            self.id_counter += 1

            self.embeddings.append(emb)
            self.hybrid_vectors.append(hybrid_vec)
            self.metadata.append({
                "id": entry_id,
                "text": text,
                "abstraction": abstraction_score
            })
            print(f"   [Pre-fill] ID {entry_id} geladen: '{text}' (Score: {abstraction_score:.3f})")

        print(f"✅ Kaltstart abgeschlossen. {len(self.metadata)} Einträge erfolgreich indiziert.\n")


    async def calculate_api_abstraction_score2(self, text):
        """Berechnet die Profilrelevanz auf einer Skala von 0 bis 10."""
        prompt = (
            f"Task: Rate the PROFILE RELEVANCE of the following statement on a scale from 0 to 10.\n\n"
            f"Guidelines:\n"
            f"- 10: Crucial long-term profile data, core identity, permanent biome/base, or fixed job description.\n"
            f"  (e.g., 'The bot lives in an area with plenty of forest', 'The bot is a farmer and harvests wheat daily')\n"
            f"- 5: Semi-permanent states, long-term plans, strategic thoughts, or repeating routines that might change.\n"
            f"  (e.g., 'The bot wants to optimize its strategy', 'The bot plans to build a secondary base later')\n"
            f"- 0: Completely fleeting, temporary event, immediate physical action, or short-term sensory observation.\n"
            f"  (e.g., 'The bot mined 5 oak logs', 'The bot can see a forest nearby right now', 'The bot sorted the chest')\n\n"
            f"Statement to analyze:\n"
            f"\"{text}\"\n\n"
            f"Respond with EXACTLY a single integer from 0 to 10. Do not write any other text.\n"
            f"Score:"
        )

        try:
            response = await client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2,
                temperature=0.3,
            )
            output = response.choices[0].message.content.strip()
            match = re.search(r'\d+', output)
            if match:
                score_val = int(match.group(0))
                return round(min(max(score_val / 10.0, 0.0), 1.0), 3)
            return 0.0
        except Exception as e:
            print(f"⚠️ Teacher-API Fehler: {e}")
            return 0.5


    def _get_openrouter_embedding(self, text, input_type="passage"):
        """Holt das 2048D Llama-Nemotron-Embedding über OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.embedding_model,
            "input": text,
            "encoding_format": "float",
            "user_metadata": {"input_type": input_type}
        }
        response = requests.post(self.openrouter_url, headers=headers, json=payload)
        response.raise_for_status()
        res_data = response.json()
        return np.array(res_data["data"][0]["embedding"], dtype=np.float32)

    def _delete_entries_by_ids(self, ids_to_delete):
        """Hilfsfunktion zum physischen Löschen von Einträgen aus allen Listen."""
        indices_to_remove = [i for i, meta in enumerate(self.metadata) if meta["id"] in ids_to_delete]
        # Von hinten nach vorne löschen, um Indexverschiebungen zu vermeiden
        for idx in sorted(indices_to_remove, reverse=True):
            self.embeddings.pop(idx)
            self.hybrid_vectors.pop(idx)
            deleted = self.metadata.pop(idx)
            print(f"   [System] Gelöscht: ID {deleted['id']} - '{deleted['text']}'")

    async def _insert_raw_entry(self, text, precalculated_emb=None, precalculated_score=None):
        """Interne Methode, um einen Eintrag final in die Listen zu schreiben."""
        if precalculated_score is None:
            score_task = self.calculate_api_abstraction_score(text)
            loop = asyncio.get_running_loop()
            embedding_task = loop.run_in_executor(None, self._get_openrouter_embedding, text, "passage")
            abstraction_score, emb = await asyncio.gather(score_task, embedding_task)
        else:
            emb = precalculated_emb
            abstraction_score = precalculated_score

        hybrid_vec = np.append(emb, abstraction_score)

        entry_id = self.id_counter
        self.id_counter += 1

        self.embeddings.append(emb)
        self.hybrid_vectors.append(hybrid_vec)
        self.metadata.append({
            "id": entry_id,
            "text": text,
            "abstraction": abstraction_score
        })
        print(f"   [System] Hinzugefügt: ID {entry_id} - '{text}' (Score: {abstraction_score:.3f})")
        return entry_id

    async def add_and_consolidate_entry(self, text, abstraction_score):
        """
        Der finale Szenen-Workflow mit DELETE-Option:
        1. Schreibt den neuen Eintrag SOFORT roh in die DB.
        2. Holt das passende semantische Cluster (die "Szene"), inklusive des neuen Eintrags.
        3. Der Agent modifiziert die Szene frei über IGNORE, REPLACE, CONSOLIDATE oder DELETE.
        """
        print(f"\n⚡ Ingestion: Schreibe neuen Eintrag direkt in die DB: '{text}'")

        # 1. SCHRITT: Sofort roh in die Datenbank einfügen
        emb = self._get_openrouter_embedding(text, "passage")
        new_entry_id = await self._insert_raw_entry(text, precalculated_emb=emb, precalculated_score=abstraction_score)

        # 2. SCHRITT: Cluster-Retrieval (Der neue Eintrag zieht seine semantischen Nachbarn an)
        retrieved_cluster = []
        if len(self.embeddings) >= 3:
            try:
                embedding_3d, soft_memberships, hdb = calculate_current_clustering(self.embeddings, self.metadata)
                retrieved_cluster = self.query_associative_memory_by_vector(emb, embedding_3d, soft_memberships)
            except Exception as e:
                print(f"  [Retrieval Fallback] Clustering fehlgeschlagen ({e}), nutze Kosinus-Ähnlichkeit...")
                retrieved_cluster = self._fallback_cosine_retrieval(emb, k=3)
        else:
            retrieved_cluster = self._fallback_cosine_retrieval(emb, k=2)

        # Sicherstellen, dass der neue Eintrag für den Agenten in der Szene sichtbar ist
        if not any(m["id"] == new_entry_id for m in retrieved_cluster):
            new_meta = next(m for m in self.metadata if m["id"] == new_entry_id)
            retrieved_cluster.append(new_meta)

        # 3. SCHRITT: Agenten auf der fertigen "Szene" operieren lassen
        print(f"  [Agent] Analysiere die aktuelle Szene mit {len(retrieved_cluster)} Einträgen...")
        decision = await self.ask_consolidation_agent(text, retrieved_cluster)
        action = decision.get("action", "IGNORE").upper()
        reason = decision.get("reason", "")
        print(chalk.blue(f"  [Agent Entscheidung] ➔ Aktion für die Szene: {action} | Grund: {reason}"))

        # 4. SCHRITT: Freie Szenen-Bearbeitung umsetzen
        if action == "IGNORE":
            # Der neue Eintrag bringt keinen Mehrwert -> Wird wieder entfernt.
            print(f"   [System] Agent verwirft den neuen Eintrag. Lösche ID {new_entry_id}...")
            self._delete_entries_by_ids([new_entry_id])

        elif action == "DELETE":
            # Ein oder mehrere alte Einträge aus der Szene sollen ersatzlos gelöscht werden.
            # Wir unterstützen hier flexibel entweder eine einzelne 'target_id' oder eine Liste 'ids_to_delete'.
            target_id = decision.get("target_id")
            ids_to_delete = decision.get("ids_to_delete", [])

            # Falls der Agent nur target_id geschickt hat, packen wir sie in die Liste
            if target_id is not None and target_id not in ids_to_delete:
                ids_to_delete.append(target_id)

            # Verhindern, dass der Agent aus Versehen den gerade geschriebenen Eintrag via DELETE killt,
            # dafür ist eigentlich IGNORE da (es sei denn, er will es explizit).
            if ids_to_delete:
                print(f"   [System] Szene bereinigt: Lösche Einträge ersatzlos: {ids_to_delete}")
                self._delete_entries_by_ids(ids_to_delete)
            else:
                print("   ⚠️ DELETE gewählt, aber keine IDs zum Löschen geliefert.")

        elif action == "REPLACE":
            # 1:1 Austausch innerhalb der Szene
            target_id = decision.get("target_id")
            consolidated_text = decision.get("consolidated_text")
            agent_score = decision.get("new_abstraction_score")

            if target_id is not None and target_id != new_entry_id:
                print(f"   [System] Szene bereinigt: Lösche veralteten Eintrag ID {target_id}.")
                self._delete_entries_by_ids([target_id])

                # Falls der Agent den Text/Score beim Ersetzen verfeinert hat,
                # updaten wir den soeben erstellten neuen Eintrag.
                if consolidated_text:
                    print(f"   [System] Überschreibe rohen Eintrag ID {new_entry_id} mit verfeinertem REPLACE-Text.")
                    # Erst den rohen Eintrag löschen und verfeinert neu einfügen
                    self._delete_entries_by_ids([new_entry_id])

                    if agent_score is not None:
                        loop = asyncio.get_running_loop()
                        c_emb = await loop.run_in_executor(None, self._get_openrouter_embedding, consolidated_text,
                                                           "passage")
                        await self._insert_raw_entry(consolidated_text, precalculated_emb=c_emb,
                                                     precalculated_score=float(agent_score))
                    else:
                        await self._insert_raw_entry(consolidated_text, precalculated_score=abstraction_score)
            else:
                print(f"   ⚠️ REPLACE ungültig (target_id fehlt oder entspricht dem neuen Eintrag).")

        elif action == "CONSOLIDATE":
            # Freies Zusammenfassen: IDs löschen (inklusive oder exklusive des neuen Eintrags) und Synthese schreiben
            ids_to_delete = decision.get("ids_to_delete", [])
            consolidated_text = decision.get("consolidated_text")
            agent_score = decision.get("new_abstraction_score")

            if ids_to_delete and consolidated_text:
                print(f"   [System] Konsolidiere Szene. Lösche Fragmente: {ids_to_delete}")
                self._delete_entries_by_ids(ids_to_delete)

                if agent_score is not None:
                    loop = asyncio.get_running_loop()
                    c_emb = await loop.run_in_executor(None, self._get_openrouter_embedding, consolidated_text,
                                                       "passage")
                    await self._insert_raw_entry(consolidated_text, precalculated_emb=c_emb,
                                                 precalculated_score=float(agent_score))
                else:
                    await self._insert_raw_entry(consolidated_text)
            else:
                print("   ⚠️ CONSOLIDATE unvollständig. Keine Änderungen an der Szene vorgenommen.")

        # Rückgabe der modifizierten Szene fürs Logging
        if retrieved_cluster:
            formatted_retrieved = ""
            for entry in retrieved_cluster:
                still_exists = any(m["id"] == entry["id"] for m in self.metadata)
                status = "AKTIV" if still_exists else "GELÖSCHT"
                formatted_retrieved += f"- [ID: {entry['id']} | {status}] \"{entry['text']}\" (Abstr.: {entry['abstraction']:.3f})\n"
            return formatted_retrieved
        return None


    def _fallback_cosine_retrieval(self, query_emb, k=3):
        """Kosinus-Ähnlichkeit als Fallback, falls noch kein HDBSCAN läuft."""
        if not self.embeddings:
            return []
        norm_embeddings = np.array(self.embeddings) / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(norm_embeddings, norm_query)

        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_k_indices:
            # Formatieren für den Consolidation-Agenten
            results.append(self.metadata[idx])
        return results

    def query_associative_memory_by_vector(self, query_emb, embedding_3d, soft_memberships):
        """
        Assoziatives Cluster-Retrieval mit adaptivem, relativem Schwellenwert.
        Verhindert das "Herausfallen" von Grenzpunkten im Mehrländereck.
        """
        rel_threshold = 0.3

        if len(self.embeddings) == 0:
            return []

        # 1. Kosinus-Ähnlichkeit zum Finden des direkten Einstiegspunkts (Best Match)
        norm_embeddings = np.array(self.embeddings) / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(norm_embeddings, norm_query)

        best_idx = np.argmax(similarities)

        # 2. Welchem Cluster gehört dieser Einstiegspunkt am stärksten an?
        best_point_memberships = soft_memberships[best_idx]
        main_cluster_idx = np.argmax(best_point_memberships)

        # Sicherheitsnetz: Wenn der Punkt zu gar keinem Cluster gehört (reines Rauschen)
        if np.sum(best_point_memberships) < 1e-5:
            return [self.metadata[best_idx]]

        results = []
        for idx, metadata in enumerate(self.metadata):
            # Zugehörigkeit des aktuellen Punkts zum Ziel-Cluster
            belonging_to_target = soft_memberships[idx, main_cluster_idx]

            # Die maximale Zugehörigkeit, die dieser Punkt überhaupt zu IRGENDEINEM Cluster hat
            max_own_belonging = np.max(soft_memberships[idx])

            if max_own_belonging < 1e-5:
                continue

            # ADAPTIVER ABGLEICH:
            # Ein Punkt wird aufgenommen, wenn seine Zugehörigkeit zum Zielcluster
            # mindestens 50% (rel_threshold=0.5) seiner eigenen maximalen Zugehörigkeit beträgt.
            if belonging_to_target >= (max_own_belonging * rel_threshold):
                results.append(metadata)

        # Fallback: Falls die adaptive Suche (warum auch immer) komplett leer bleibt,
        # liefere zumindest den direkten Kosinus-Nachbarn zurück.
        if not results:
            results = [self.metadata[best_idx]]

        return results

    async def ask_consolidation_agent(self, new_entry, retrieved_entries):
        """Befragt den Agenten nach der optimalen Integrations-Strategie und liefert globalen Abstraktions-Kontext."""
        # 1. Berechne globale Abstraktions-Statistiken für den Agenten
        if self.metadata:
            all_scores = [m["abstraction"] for m in self.metadata]
            global_stats = (
                f"Total memories in database: {len(self.metadata)}\n"
                f"Global Abstraction Scores -> Min: {min(all_scores):.2f}, "
                f"Max: {max(all_scores):.2f}, "
                f"Average: {sum(all_scores) / len(all_scores):.2f}"
            )
        else:
            global_stats = "Global Abstraction Scores -> Database is currently empty."

        yaml_path = "consolidation/reason_info3.yaml"

        # 2. Formatiere die retrieved Einträge
        formatted_retrieved = ""
        for entry in retrieved_entries:
            formatted_retrieved += f"- [ID: {entry['id']}] \"{entry['text']}\" (Abstr.: {entry['abstraction']:.3f})\n"

        if not os.path.exists(yaml_path):
            print(f"❌ Prompt-Datei nicht gefunden unter: {yaml_path}")
            return {}

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Fehler beim Laden der YAML-Datei: {e}")
            return {}

        base_prompt = config["messages"][0]["content"]
        full_prompt = base_prompt.replace("{{$memory_fragments}}", formatted_retrieved)

        # print(f"Prompt: {prompt}")

        try:
            response = await client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"⚠️ Fehler im Consolidation Agent: {e}")
            return {"action": "ADD", "reason": "Fallback due to API error", "target_id": None, "ids_to_delete": [],
                    "consolidated_text": None, "new_abstraction_score": None}

    def save(self, bot_name):
        base_path = f"bots/{bot_name}/database/hierarchical_memory"
        np.save(f"{base_path}_embeddings.npy", np.array(self.embeddings))
        np.save(f"{base_path}_hybrid_vectors.npy", np.array(self.hybrid_vectors))
        with open(f"{base_path}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
        print(f"\n[Info] Daten erfolgreich gespeichert unter '{base_path}_*'.")

    def load(self, bot_name):
        """
        Versucht, die gespeicherten Vektoren und Metadaten zu laden.
        Gibt True zurück, wenn der Import erfolgreich war, andernfalls False.
        """

        base_path=f"bots/{bot_name}/database/hierarchical_memory"

        emb_path = f"{base_path}_embeddings.npy"
        hyb_path = f"{base_path}_hybrid_vectors.npy"
        meta_path = f"{base_path}_metadata.json"

        if os.path.exists(emb_path) and os.path.exists(hyb_path) and os.path.exists(meta_path):
            try:
                # 1. Metadaten einlesen
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                # 2. NumPy-Arrays einlesen und in Listen aus Arrays konvertieren
                loaded_embeddings = np.load(emb_path)
                loaded_hybrid = np.load(hyb_path)

                self.embeddings = [arr for arr in loaded_embeddings]
                self.hybrid_vectors = [arr for arr in loaded_hybrid]

                # 3. ID Counter anhand des höchsten gelesenen IDs inkrementieren
                if self.metadata:
                    self.id_counter = max(item["id"] for item in self.metadata) + 1
                else:
                    self.id_counter = 0

                print(
                    f"📂 [Memory] Speicher erfolgreich aus '{base_path}_*' wiederhergestellt. ({len(self.metadata)} Einträge)")
                return True
            except Exception as e:
                print(f"⚠️ Fehler beim Laden der Speicherdateien: {e}. Starte mit neuem Speicher.")
                return False
        return False


def calculate_current_clustering(embeddings, metadata, min_cluster_size=2, target_influence=0.002):
    """
    Berechnet das Poincaré-Layout mit hoher numerischer Präzision (float64).
    Löst feinste Nuancen nahe der maximalen Abstraktion (1.0) asymptotisch auf.
    """
    # 1. Erzwinge durchgehend float64-Präzision
    embeddings = np.array(embeddings, dtype=np.float64)

    # Abstraktionswerte hochpräzise extrahieren
    abstractions = np.array([m["abstraction"] for m in metadata], dtype=np.float64)

    # Wenn zu wenig Daten da sind, bauen wir eine Pseudo-Zugehörigkeitsmatrix
    if len(embeddings) < 3:
        return np.zeros((len(embeddings), 3)), np.ones((len(embeddings), 1))

    n_pts = len(embeddings)

    # 2. Semantische 3D-Projektion (UMAP)
    dynamic_neighbors = max(2, min(4, n_pts - 1))

    # Für Clustering: mehr Dimensionen behalten
    reducer_cluster = umap.UMAP(
        n_neighbors=dynamic_neighbors,
        min_dist=0.0,
        n_components=min(10, n_pts - 2),
        metric='cosine',
        init='random',
        low_memory=False,
        random_state=42
    )

    raw_projection_cluster = reducer_cluster.fit_transform(embeddings).astype(np.float64)

    # Für die Visualisierung separat auf 3D reduzieren (nur fürs Plotten)
    reducer_viz = umap.UMAP(
        n_neighbors=dynamic_neighbors,
        min_dist=0.0,
        n_components=3,
        metric='cosine',
        init='random',
        low_memory=False,
        random_state=42
    )
    raw_projection_viz = reducer_viz.fit_transform(embeddings).astype(np.float64)

    print(f"[Poincaré-Ball 3D] Berechne semantische Richtungen...")

    # Mathematisch stabiles, asymptotisches Schrumpfen des Radius:
    # Je näher die Abstraktion an 1.0 rückt, desto dramatischer schrumpft r.
    #
    # WICHTIG: min_radius_frac verhindert, dass Punkte mit abstraction=1.0
    # exakt auf (0,0,0) kollabieren. Ohne Floor geht bei r=0 jede Richtungs-
    # information (aus UMAP) verloren -> zwei völlig unterschiedliche
    # "Kern-Identitäten" (z.B. "ist Miner" vs. "ist Farmer") würden sonst auf
    # demselben Punkt landen, obwohl sie semantisch klar unterscheidbar sind.
    #
    # min_radius_frac=0.0  -> altes Verhalten (r kann exakt 0 werden)
    # min_radius_frac=0.03 -> r bleibt selbst bei abstraction=1.0 minimal > 0,
    #                         Richtung/Winkel bleibt also erhalten
    min_radius_frac = 0.03

    # Radius-Formel gilt für beide Räume gleichermaßen (hängt nur vom Abstraktionsscore ab)
    gamma = 3.5
    abstractions_clipped = np.clip(abstractions, 0.0, 1.0 - 1e-12)
    shrink_term = 1.0 - np.power(abstractions_clipped, gamma)
    radii = 0.98 * shrink_term * (1.0 - min_radius_frac) + min_radius_frac

    # Clustering-Raum: Richtung * Radius
    norms_c = np.linalg.norm(raw_projection_cluster, axis=1, keepdims=True)
    norms_c[norms_c == 0] = 1e-15
    direction_cluster = raw_projection_cluster / norms_c
    cluster_points = direction_cluster * radii[:, np.newaxis]

    # Viz-Raum (3D): Richtung * Radius -- das ist der Rückgabewert fürs Plotten
    norms_v = np.linalg.norm(raw_projection_viz, axis=1, keepdims=True)
    norms_v[norms_v == 0] = 1e-15
    direction_viz = raw_projection_viz / norms_v
    embedding_3d = direction_viz * radii[:, np.newaxis]

    dist_matrix = poincare_distance_matrix(cluster_points)

    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        allow_single_cluster=False,
        cluster_selection_method='leaf',
        metric='precomputed',  # <-- statt Standard-euklidisch
        copy=True
    )
    hdb.fit(dist_matrix)

    # 6. Soft-Clustering via Distanz-Softmax
    unique_labels = [l for l in np.unique(hdb.labels_) if l != -1]
    n_clusters = len(unique_labels)
    soft_memberships = np.zeros((len(embeddings), max(1, n_clusters)))

    if n_clusters > 0:
        centroids = []
        for label in unique_labels:
            pts = cluster_points[hdb.labels_ == label]
            centroids.append(pts.mean(axis=0))
        centroids = np.array(centroids)

        temperature = 0.1
        # centroids berechnen wie bisher (Mittelwert im Ball ist zwar auch nur eine
        # euklidische Näherung an den echten "Fréchet-Mittelpunkt", aber für die
        # Softmax-Gewichtung reicht das i.d.R. aus)
        for idx in range(len(embeddings)):
            point = cluster_points[idx]
            abs_val = abstractions[idx]
            dists = np.array([
                poincare_distance_matrix(np.vstack([point, c]))[0, 1]
                for c in centroids
            ])
            point_temperature = temperature * (1.0 + abs_val) ** 6
            exp_dists = np.exp(-dists / point_temperature)
            soft_memberships[idx] = exp_dists / np.sum(exp_dists)
    else:
        soft_memberships = np.ones((len(embeddings), 1))

    print(f"[HDBSCAN] {n_clusters} natürliche Cluster identifiziert.")
    print(f"[Debug] Harte Labels: {list(hdb.labels_)}")

    return embedding_3d, soft_memberships, hdb


def visualize_final_memory(bot_name, memory, min_cluster_size=2, target_influence=0.002):
    """Visualisiert das Gedächtnis innerhalb einer dreidimensionalen Poincaré-Drahtgitterkugel."""

    rel_threshold = 0.5

    if len(memory.embeddings) < 3:
        print("[Visualisierung] Zu wenig Daten zum Plotten.")
        return

    embedding_3d, soft_memberships, hdb = calculate_current_clustering(
        memory.embeddings, memory.metadata, min_cluster_size, target_influence
    )

    metadata = memory.metadata
    n_clusters = soft_memberships.shape[1]
    cluster_base_colors = generate_colors(n_clusters)
    noise_color = np.array([120, 120, 120])
    abstractions = np.array([m["abstraction"] for m in metadata])

    colors = []
    hover_texts = []

    print("\n--- DEBUG: Abstraktion & Radius pro Punkt ---")
    for i, m in enumerate(metadata):
        r = np.linalg.norm(embedding_3d[i])
        print(f"  ID {m['id']:>2} | abstraction={m['abstraction']:.3f} | radius={r:.4f} | text='{m['text'][:40]}'")
    print(f"\nRadius-Range: min={np.linalg.norm(embedding_3d, axis=1).min():.4f}  "
          f"max={np.linalg.norm(embedding_3d, axis=1).max():.4f}")


    for i in range(len(metadata)):
        p_norm = soft_memberships[i]
        max_own_belonging = np.max(p_norm)

        active_indices = []
        active_probs = []
        for j in range(n_clusters):
            prob = p_norm[j]
            if prob >= (max_own_belonging * rel_threshold) and max_own_belonging > 1e-5:
                active_indices.append(j)
                active_probs.append(prob)

        if hdb.labels_[i] == -1 and not active_indices:
            r, g, b = noise_color
        else:
            active_probs = np.array(active_probs)
            sum_active = np.sum(active_probs)
            if sum_active > 1e-5:
                normalized_probs = active_probs / sum_active
                mixed_rgb = np.zeros(3)
                for idx, cluster_idx in enumerate(active_indices):
                    mixed_rgb += normalized_probs[idx] * cluster_base_colors[cluster_idx]
                r, g, b = int(mixed_rgb[0]), int(mixed_rgb[1]), int(mixed_rgb[2])
            else:
                r, g, b = noise_color

        colors.append(f"rgb({r}, {g}, {b})")

        # --- HOCHPRÄZISE TEXT-ANZEIGE FÜR DEN HOVER-TOOLTIP ---
        belonging_lines = []
        for j in range(n_clusters):
            prob = p_norm[j]
            is_associative = j in active_indices
            assoc_star = " ⭐" if is_associative else ""
            is_main = (j == np.argmax(p_norm))
            prefix = "<b>" if is_main else ""
            suffix = "</b>" if is_main else ""
            # Höhere Präzision für die Cluster-Zugehörigkeit (3 Nachkommastellen)
            belonging_lines.append(f"{prefix}● Cluster {j + 1}: {prob * 100:.3f}%{assoc_star}{suffix}")

        belonging_str = "<br>".join(belonging_lines)

        # Abstraktion und Radius werden jetzt mit 6 Nachkommastellen (.6f) gerendert,
        # damit Unterschiede wie 0.999900 und 0.999999 visuell sofort trennscharf sind!
        text_content = (
            f"<b>Eintrag (ID {metadata[i]['id']}):</b> '{metadata[i]['text'][:200]}'<br>"
            f"<b>Abstraktionsgrad:</b> {abstractions[i]:.6f}<br>"
            f"<b>Position im Ball:</b> Radius r = {np.linalg.norm(embedding_3d[i]):.6f}<br>"
            f"<b>Zugehörigkeiten:</b><br>{belonging_str}"
        )
        hover_texts.append(text_content)

    marker_sizes = 12 + 18 * abstractions

    fig = go.Figure()

    # --- HINZUFÜGEN DES TRANSPARENTEN 3D-KUGELGITTER-DRAHTMODELLS (Poincaré-Kugel) ---
    # Wir zeichnen Breiten- und Längengrade für die 3D-Visualisierung
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)

    # Gitternetz-Linien für Längengrade (vertikal)
    for phi in u[::4]:  # Jede vierte Linie zeichnen, um das Gitter übersichtlich zu halten
        xs = 0.95 * np.cos(phi) * np.sin(v)
        ys = 0.95 * np.sin(phi) * np.sin(v)
        zs = 0.95 * np.cos(v)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.08)', width=1),
            showlegend=False, hoverinfo='none'
        ))

    # Gitternetz-Linien für Breitengrade (horizontal)
    for theta in v[::3]:
        xs = 0.95 * np.cos(u) * np.sin(theta)
        ys = 0.95 * np.sin(u) * np.sin(theta)
        zs = 0.95 * np.ones_like(u) * np.cos(theta)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.08)', width=1),
            showlegend=False, hoverinfo='none'
        ))

    # Die echten Datenpunkte
    fig.add_trace(go.Scatter3d(
        x=embedding_3d[:, 0],
        y=embedding_3d[:, 1],
        z=embedding_3d[:, 2],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=colors,
            opacity=0.9,
            line=dict(width=1, color='white')
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Erinnerungen'
    ))

    fig.update_layout(
        title=(
            "Echter 3D Poincaré-Ball-Vektorspeicher<br>"
            "<sup>Zentrum (r=0) = Maximale Abstraktion | Äußere Kugelschale (r=1) = Konkretes Rauschen</sup>"
        ),
        scene=dict(
            xaxis=dict(title="X", range=[-1.0, 1.0]),
            yaxis=dict(title="Y", range=[-1.0, 1.0]),
            zaxis=dict(title="Z", range=[-1.0, 1.0]),
            aspectmode='cube'
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=80)
    )

    # --- DIAGRAMM AUTOMATISCH ALS HTML SPEICHERN ---
    # Wir fügen den optionalen Parameter `save_dir` hinzu (siehe Schritt 2 für den Funktionskopf)
    save_dir = f"bots/{bot_name}/Memory_History"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/poincare_ball_{timestamp}.html"

        # Sichert das komplette, interaktive Plotly-Diagramm als HTML-Datei
        fig.write_html(filename, auto_open=False)
        print(f"💾 [Visualisierung] Interaktiver Plot gespeichert unter: {filename}")


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(np.array([int(r * 255), int(g * 255), int(b * 255)]))
    return colors


def poincare_distance_matrix(points, eps=1e-9):
    """
    points: (N, D) Array, alle mit Norm < 1 (im offenen Ball)
    Gibt eine (N, N) Poincaré-Distanzmatrix zurück.
    """
    norms_sq = np.sum(points**2, axis=1)
    norms_sq = np.clip(norms_sq, 0, 1 - eps)  # numerische Sicherheit

    diff = points[:, None, :] - points[None, :, :]
    sq_dists = np.sum(diff**2, axis=-1)

    denom = (1 - norms_sq)[:, None] * (1 - norms_sq)[None, :]
    denom = np.clip(denom, eps, None)

    arg = 1 + 2 * sq_dists / denom
    arg = np.clip(arg, 1.0 + eps, None)  # arccosh braucht arg >= 1

    return np.arccosh(arg)
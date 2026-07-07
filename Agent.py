import multiprocessing
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from simple_chalk import chalk
from langchain_openai import ChatOpenAI
import asyncio
import json
import os
from pathlib import Path
import argparse
import logging

from Loader import write_log
from RAG import generate_rag_query
from consolidation.run_reasoning import run_memory_pipeline
from Amygdala import Amygdala
import base64


db_instance = None

# Tools
class ActionType(str, Enum):
    EAT = "eat"
    DIG = "dig"
    CRAFT = "craft"
    CHAT = "chat"
    DEPOSIT = "putInChest"
    WITHDRAW = "takeFromChest"
    VIEW = "viewChest"
    SMELT = "smeltItem"
    CLEAR = "clearFurnace"
    Go = "goToCoordinates"
    #PLACE = "placeHere"
    #SCAN = "scanBlocks"
    FARM = "doFarming"
    ATTACK = "attackMob"
    NONE = "none"


class MinecraftAction(BaseModel):
    """The structured format for bot decision-making."""
    action: ActionType = Field(description="Kind of tool.")
    item: Optional[str] = Field(None, description="The item (e.g. 'berries') or none.")
    count: Optional[int] = Field(default=1, description="How many items...")

    x: Optional[float] = Field(None, description="Only required for goToCoordinates.")
    y: Optional[float] = Field(None, description="Only required for goToCoordinates.")
    z: Optional[float] = Field(None, description="Only required for goToCoordinates.")

    reasoning: str = Field(
        description="Description of your reasoning, if you use a tool. "
                    "Also write here a specific overarching goal that should be achieved with the tool use. "
                    "If you use chat, write e.g. 'Hello, what do we want to do?', "
                    "'What tasks do you want to complete?' etc."
    )

    search_intent: Optional[str] = Field(
            None,
            description="A rough, direct question or explicit keywords about what historical memories, "
                        "coordinates, or technical knowledge you need right now to back up this action "
                        "(e.g., 'Where is the melon field?', 'How did I build the automated furnace?')."
    )


class SimpleMemory:
    """A consolidation system with AI-based summarization."""

    def __init__(self, bot_name, logger, max_messages: int = 5, summarizer_llm=None):
        self.action_data = ""
        self.search_intent = ""
        self.max_messages = max_messages
        self.summary = ""
        self.bot_name = bot_name

        self.bot_dir = f"bots/{bot_name}/memory"
        os.makedirs(self.bot_dir, exist_ok=True)

        self.persistence_file = f"{self.bot_dir}/{bot_name}_memory.json"
        self.sessions_file = f"{self.bot_dir}/sessions.jsonl"

        # During initialization, attempt to load old consolidation.
        self.load_from_disk(logger)


    def load_from_disk(self, logger):
        """Loads consolidation from disc."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.summary = data.get("summary", "")
                logger.info(f"💾 Loaded consolidation for {self.bot_name}.")
            except Exception as e:
                print(f"⚠️ Error while loading: {e}")


    def save_to_disk(self):
        """Saves the consolidation."""
        data = {"summary": self.summary}
        with open(self.persistence_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def summarize_old_messages(self, observation_time, observation, logger):
        """Adds an input-output pair to consolidation."""

        self.summary = f"Last action: {self.action_data} \n Last search intent: {self.search_intent} \n Last feedback: {observation.get("Tool feedback", "")}"

        logger.info(f"✅ Memory: {self.summary[:500]}...")

        # Write to sessions.jsonl for PREMem (append mode)
        session_entry = {
            "session_id": f"session_{int(time.time())}",
            "session_date": observation_time,
            "text": self.summary
        }

        with open(self.sessions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_entry, ensure_ascii=False) + "\n")

        logger.info(f"✅ New summary saved in sessions.jsonl and local consolidation updated.")

        self.save_to_disk()


    def get_formatted_history(self, log_path, logger) -> str:
        """Creates a readable structure for the system prompt."""
        formatted_parts = []
        final_summary = []

        # 1. Long term consolidation (summary)
        if self.summary:
            formatted_parts.append("### LONG-TERM MEMORY (Summary of past events)")
            formatted_parts.append(self.summary)
            formatted_parts.append("-" * 40)

            final_summary.append("### LONG-TERM MEMORY (Summary of past events)")
            final_summary.append(self.summary)
            final_summary.append("-" * 40)

        # If consolidation is empty
        if not formatted_parts:
            return "No previous interactions recorded."

        # Join together to form a clean block
        final_history = "\n".join(formatted_parts)
        final_summary2 = "\n".join(final_summary)

        write_log(log_path, final_history)

        # Display the debug log nicely in the console
        logger.info(chalk.blue("\n--- CONTEXT SENT TO AI ---"))
        logger.info(final_history)
        logger.info(chalk.blue("--------------------------\n"))

        return final_summary2

    def observation_llm(self, observation):

        final_output = {
            f"# LAST ACTION"
            f"action {self.action_data} {self.search_intent}"
            f"feedback {observation.get("Tool feedback", "")}\n"
            
            f"# CURRENT OBSERVATION"
            f"time {observation.get("time", "")}"
            f"food {observation.get("food", "")}"
            f"inventory {observation.get("inventory", "")}"
            f"{observation.get("Armor slots", "")}"
            f"surroundings {observation.get("surroundings", "")}\n"
            
            f"# WORLDCHAT"
            f"{observation.get("entities", "")}"
            f"{observation.get("Chat history", "")}"
        }

        return final_output

# --- PRIVATE QUEUES & PROZESS-VAR ---
_request_queue = None
_response_queue = None

amy = Amygdala()


# --- WORKER PROZESS ---
def agent_worker_process(req_q, res_q, log_path, bot_name, embed_req_q, embed_res_q):
    """Hintergrundprozess für die API-Kommunikation mit Gedächtnis."""
    print(f"{bot_name}: ✅ API Agent gestartet. Warte auf Observations...")

    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    success = load_dotenv(dotenv_path=env_path)

    if not success:
        print(f"❌ Kritisch: .env konnte nicht geladen werden unter {env_path}")

    logger = get_bot_logger(bot_name)

    # The Memory-Objekt
    memory = SimpleMemory(logger=logger, bot_name=bot_name, max_messages=5)

    args = parse_arguments()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    json_schema = MinecraftAction.model_json_schema()

    # DEFINITION OF THE SYSTEM PROMPT
    system_prompt = (
        f"Your name is '{bot_name}'. You are a bot living in a Minecraft world. "
        "You ensure that resources are processed correctly. "
        "You are also communicative and coordinate with other bots to manage resource extraction, but also discuss metacognitive topics."
        
        "You can communicate and interact with the game by using all available commands. "
        "Always respond in a concise 1-2 sentence format, followed by a command to execute your action. "
        "Examples for this are provided below. NEVER try using a command that doesn't exist! "
        "If you receive a message from 'System', treat it as an automated event and respond as if you had the thought yourself. "
        "Use it as an opportunity to appear more lively or take initiative."
        f"Output ONLY valid JSON matching this schema: {json_schema}. "
        
        "\n### TOOL DEFINITIONS:\n"
        "- smeltItem: Starts the smelting process. You must have fuel (coal) and the item in your inventory.\n"
        "- clearFurnace: Use this to retrieve finished ingots. Only use this if you know items are ready or the furnace stopped burning.\n"
        "- dig: Used to gather resources like oak_log or stone. Specify the block name. You can only dig 1 block at a time.\n"
        "- goToCoordinates: Use this for travel to known locations from your consolidation.\n"
        "- craft: Craft the given recipe a given number of times.\n"
        "- eat: Eat/drink the given item.\n"
        "- chat: Add a message in world chat.\n"
        "- putInChest: Put the given item in the nearest chest.\n"
        "- takeFromChest: Take the given items from the nearest chest.\n"
        "- viewChest: View the items/counts of the nearest chest.\n"
        #"- placeHere: Place a given block at a location.\n"
        #"- scanBlocks: Performs a high-precision coordinate scan of the immediate 10x10x10 area. "
        #    "Use the placeBlock Tool if you want to place a block afterwards.\n"
        "- doFarming: Wheat is harvested from the nearest ripe field."
        "- attackMob: Attack a specific mob"

        "\n### EXAMPLE:\n"
        "{\n"
        '  "action": "dig",\n'
        '  "item": "oak_log",\n'
        '  "count": 10,\n'
        '  "reasoning": "I need more oak logs to build a shelter. I will gather 10 more to increase my inventory."\n'
        "}\n"

        "{\n"
        '  "action": "smeltItem",\n'
        '  "item": "raw_copper",\n'
        '  "count": 10,\n'
        '  "reasoning": "I see a furnace and I have raw copper and coal in my inventory. I should smelt it to get copper ingots."\n'
        "}\n"

        "{\n"
        '  "action": "goToCoordinates",\n'
        '  "x": 125.5,\n'
        '  "y": 64,\n'
        '  "z": -300.2,\n'
        '  "reasoning": "I am heading back to the coordinates where I previously found a large coal vein to continue mining."\n'
        "}\n"

        "{\n"
        '  "action": "clearFurnace",\n'
        '  "reasoning": "I have waited long enough for the copper to smelt. I am now checking the furnace to collect the ingots."\n'
        "}\n"
    )

    while True:
        try:
            observation = req_q.get()
            if observation is None: break

            # Subagent for Observation
            observation_response = memory.observation_llm(observation)

            history_text = memory.get_formatted_history(log_path, logger) # Loading history
            observation_time = str(observation.get("time", ""))

            rag_context = ""

            if not memory.action_data == "":
                memory.summarize_old_messages(observation_time, observation, logger) # Update history

                rag_query = generate_rag_query(observation, memory.summary)  # Generate Query
                logger.info("rag_query: " + rag_query)

                rag_context = run_memory_pipeline(
                    args=args,
                    bot_name=bot_name,
                    query_text=rag_query,  # Die RAG-Suchanfrage
                    observation=observation,  # Aktuelle Bot-Observation zwecks Zeitstempel
                    embed_req_q=None,
                    embed_res_q=None,
                    logger=logger
                )

            # EXTEND SYSTEM PROMPT WITH RAG
            system_prompt_with_rag = system_prompt
            messages = [{"role": "system", "content": system_prompt_with_rag}]

            messages.append({
                "role": "user",
                "content": f"{rag_context}"
            })

            messages.append({
                "role": "user",
                "content": f"{observation_response}"
            })

            # Call Amygdala
            warning = amy.inject_to_prompt()

            # Append warnings from Amygdala
            if warning:
                messages.append({
                    "role": "user",
                    "content": f"{warning}"
                })

            print(chalk.green(json.dumps(messages, indent=2, ensure_ascii=False)))

            response = client.chat.completions.create(
                model="z-ai/glm-5.2",
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )

            print(response)

            raw_text = response.choices[0].message.content.strip()
            raw_json_string = extract_single_json(raw_text)

            print(f"DEBUG Cleaned JSON string: [{raw_json_string}]")

            if not raw_json_string:
                raise ValueError("Could not extract a valid, isolated JSON object from the LLM response.")

            try:
                action_data = MinecraftAction.model_validate_json(raw_json_string)
                data = action_data.model_dump()

                print(f"✅ Determined tool: {action_data.action}")
                print(f"📝 Reasoning: {action_data.reasoning[:500]}...")
                print(f"🔢 Count: {action_data.count}")
                print(f"Question: {action_data.search_intent}")

                #amy.analyze_situation(data, observation)

                memory.action_data = action_data.reasoning[:500]
                memory.search_intent = action_data.search_intent[:500] if action_data.search_intent else ""

                # Daten in die Antwort-Queue legen
                res_q.put(data)
                logger.info("📤 Queue entry confirmed.")

            except Exception as val_e:
                print(f"⚠️ Validation failed: {val_e}")
                res_q.put({"action": "none", "item": None, "reasoning": "Validation failed."})

        except Exception as api_e:
            print(f"❌ API failure: {api_e}")
            import traceback
            traceback.print_exc()
            res_q.put({"action": "none", "item": None, "reasoning": f"API error: {str(api_e)}"})


def extract_single_json(raw_text: str) -> Optional[str]:
    """Extracts the first valid JSON string from a string containing noise."""

    # Remove initial and final Markdown code blocks
    if raw_text.startswith("```"):
        try:
            raw_text = raw_text.split("```")[1].replace("json", "").strip()
        except IndexError:
            pass

    # Removal of invisible characters
    raw_text = raw_text.replace('\xa0', ' ').replace('\u200b', '').strip()

    # Find first '{'
    start = -1
    try:
        start = raw_text.index('{')
    except ValueError:
        return None  # No JSON found

    # Parenthesis counting logic to find the end of the first complete JSON object
    balance = 0
    end = -1
    for i in range(start, len(raw_text)):
        char = raw_text[i]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1

        if balance == 0:
            end = i + 1
            break

    if end != -1:
        # Extract only the first complete JSON object
        json_string = raw_text[start:end].strip()

        # Final sanity check: Is it a valid JSON file?
        try:
            json.loads(json_string)
            return json_string
        except json.JSONDecodeError:
            pass

    return None  # Could not isolate valid JSON


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run episodic consolidation reasoning experiment with 10x compression.")
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o-mini")
    #parser.add_argument("--model_name", type=str, default="qwen/qwen3-235b-a22b-2507")
    #parser.add_argument("--model_name", type=str, default="z-ai/glm-5.2")
    parser.add_argument('--api_base', type=str, default='https://openrouter.ai/api/v1', help='API base URL')
    parser.add_argument("--embedding_model_name", type=str, default="NovaSearch/stella_en_400M_v5")
    parser.add_argument("--dataset_name", type=str, default="minecraft",
                        choices=["locomo", "longmemeval_s", "minecraft"])
    parser.add_argument("--mode", type=str, default="turn")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_cache_dir", type=str, default=".cache")
    parser.add_argument("--token_budget", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--show_progress_bar", type=bool, default=True)
    parser.add_argument("--compress_rate", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--cpu_workers", type=int, default=None)
    parser.add_argument("--compression_factor", type=int, default=5)
    return parser.parse_args()


def get_bot_logger(bot_name):
    """
    Erstellt einen isolierten Logger für einen Bot, der in eine eigene Datei
    und optional parallel in die Konsole schreibt.
    """
    # Verzeichnis für die Logs erstellen, falls nicht vorhanden
    log_file = f"bots/{bot_name}/detailed_{bot_name}.log"

    # Logger mit dem Namen des Bots holen (verhindert Namenskollisionen)
    logger = logging.getLogger(f"bot_{bot_name}")
    logger.setLevel(logging.INFO)

    # WICHTIG: Handler nur hinzufügen, wenn sie nicht schon existieren
    # (verhindert doppelte Log-Einträge bei mehrfachem Aufruf)
    if not logger.handlers:
        # 1. File Handler: Schreibt detailliert mit Timestamp in die Datei
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Verhindert, dass die Logs an den Root-Logger weitergegeben und dort doppelt gedruckt werden
    logger.propagate = False

    return logger


# --- PUBLIC INTERFACE FOR MAIN.PY ---

def init_queues():
    """Creates and returns the queues to be used in Main.py."""
    # Erstellt die Queues explizit im 'spawn'-Kontext, damit sie mit
    # dem spawn-Prozess kompatibel sind.
    ctx = multiprocessing.get_context('spawn')
    return ctx.Queue(), ctx.Queue()


def start_agent_process(req_q, res_q, bot_name, embed_req_q, embed_res_q):
    """Create a new process and return it."""

    bot_base_dir = f"bots/{bot_name}"
    os.makedirs(f"{bot_base_dir}", exist_ok=True)

    log_path = f"{bot_base_dir}/{bot_name}_ai.log"

    # Nutze den 'spawn'-Kontext anstelle des Standard-'fork' unter Linux
    ctx = multiprocessing.get_context('spawn')

    process = ctx.Process(
        target=agent_worker_process,
        args=(req_q, res_q, log_path, bot_name, embed_req_q, embed_res_q)  # Hier erweitert
    )
    process.start()
    print(f"🚀 New LLM worker launched via SPAWN (PID: {process.pid})")
    return process


async def fetch_agent_response(res_q):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, res_q.get)


def run_agent_async(req_q, obs):
    """Send new observation."""
    req_q.put(obs)


def stop_agent_worker(req_q):
    """It ends the process cleanly."""
    global _agent_process
    if _agent_process and _agent_process.is_alive():
        req_q.put(None)
        _agent_process.join()


__all__ = ['run_agent_async', 'fetch_agent_response', 'start_agent_process', 'stop_agent_worker', 'init_queues']
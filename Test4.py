import os
import re
import sqlite3
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from dotenv import load_dotenv
from transformers import GenerationConfig
from simple_chalk import chalk
import pickle
import torch.nn.functional as F
import asyncio
import math
from openai import AsyncOpenAI
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import statistics
from peft import prepare_model_for_kbit_training

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# CONFIGURATION & PFADE
# =====================================================================
LOCAL_MODEL_PATH = "/home/benito/PycharmProjects/Holist 2.0/models/Qwen3.5-0.8B"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PARQUET_DATA_PATH = "andy-4.1.parquet"
DB_PATH = "holist_memory.db"
OUTPUT_LOGPROBS_PATH = "collected_policy_logprobs.pkl"

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 2048
LEARNING_RATE = 1e-6  # Konservative LR für stabiles Alignment

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

TEACHER_MODEL = "meta-llama/llama-3.1-70b-instruct"


# =====================================================================
# 1. DATABASE SETUP & STATE MANAGEMENT
# =====================================================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def reset_database(conn):
    """Löscht alle Einträge, um für die nächste Epoche/Run eine saubere Baseline zu haben."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM episodic_memory")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='episodic_memory'")
    conn.commit()
    print("🧹 Datenbank erfolgreich zurückgesetzt.")


def query_database_memories(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM episodic_memory ORDER BY id ASC")
    rows = cursor.fetchall()
    return [{"id": r[0], "text": r[1]} for r in rows]


def execute_andy_sql_safely(conn, sql_command: str):
    allowed_actions = ["INSERT INTO episodic_memory", "DELETE FROM episodic_memory", "UPDATE episodic_memory"]
    if not any(action in sql_command for action in allowed_actions):
        print(f"⚠️ SQL Abgewiesen (Sicherheits-Filter): {sql_command}")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(sql_command)
        conn.commit()
        print(f"📁 SQL Erfolgreich ausgeführt: {sql_command}")
        return True
    except sqlite3.Error as e:
        print(f"❌ SQLite Fehler bei der Ausführung: {e}")
        return False


# =====================================================================
# 2. DATA PROCESSING
# =====================================================================
def extract_summary_and_prompt(conversations):
    summary, prompt = "", ""
    for msg in conversations:
        if msg["from"] == "human":
            val = msg["value"]
            match = re.search(r"Old Memory: '(.*?)'(\n|Recent conversation:|$)", val, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                prompt_part = val.split("Recent conversation:")[-1] if "Recent conversation:" in val else val
                prompt = prompt_part.strip().split("Summarize your old memory")[0].strip()
            break
    return summary, prompt


# =====================================================================
# 3. RERANKING
# =====================================================================
def get_top_k_memories(query_episode: str, memories: list, ce_model, ce_tokenizer, device, top_k: int = 5):
    if not memories:
        return ""
    pairs = [[query_episode, mem["text"]] for mem in memories]
    features = ce_tokenizer(pairs, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(
        device)

    with torch.no_grad():
        logits = ce_model(**features).logits.squeeze(-1)

    if logits.ndim == 0:
        logits = logits.unsqueeze(0)

    actual_top_k = min(top_k, len(memories))
    top_indices = torch.topk(logits, actual_top_k).indices.tolist()

    retrieved_chunks = []
    for idx in top_indices:
        mem = memories[idx]
        retrieved_chunks.append(f"[ID: {mem['id']}]: '{mem['text']}'")

    return "\n".join(retrieved_chunks)


# =====================================================================
# 4. GENERATION WITH DETACHED GRAPH COLLECTORS
# =====================================================================
def generate_with_trainable_logprobs(prompt_text, tokenizer, model, device, max_tokens=512, temp=0.3):
    """
    Generiert Text, behält aber die Logprobs als differenzierbare Tensoren im Speicher,
    damit wir später .backward() darauf aufrufen können.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    print(chalk.blue(f"Input model 1: {prompt_text}"))

    # Temporär no_grad für die Autoregressiv-Schleife, um VRAM zu sparen
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,  # Leicht erhöht für mehr Kreativität gegen leere Antworten
            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Die exakte Anzahl der Input-Token aus dem Dictionary auslesen
    actual_input_len = inputs["input_ids"].shape[1]

    generated_sequence = generation_output.sequences[0]

    # Sicher abschneiden ab der echten Länge des Prompts
    generated_tokens = generated_sequence[actual_input_len:]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Debug-Print, um zu sehen, was hier schiefläuft
    print(chalk.blue(f"Debug -> Prompt-Tokens: {actual_input_len} | Gesamt-Tokens: {len(generated_sequence)}"))
    print(chalk.yellow(f"Output model 1: {decoded_text}"))

    if len(generated_tokens) == 0:
        return decoded_text, torch.tensor(0.0, device=device, requires_grad=True)

    # RE-FORWARD PASS (nur über diese eine Sequenz) mit Gradienten-Tracking für den Loss
    full_outputs = model(generated_sequence.unsqueeze(0))
    logits = full_outputs.logits[0, input_len - 1: -1, :]  # Shift für Kausalität

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs[torch.arange(len(generated_tokens)), generated_tokens]

    # Wir nehmen den Mittelwert der Token-Logprobs als Sequenz-Score
    mean_sequence_log_prob = target_log_probs.mean()

    return decoded_text, mean_sequence_log_prob


# =====================================================================
# 5. COGNITIVE PIPELINE LOOP
# =====================================================================
def run_cognitive_loop(prompt, conn, ce_model, ce_tokenizer, tokenizer, model, device):
    # Schritt 1: Meso-Ebene (Rewrite)
    rewrite_prompt = (
        "You are an advanced memory consolidation system.\n"
        "Task: Rewrite raw logs or thoughts into a single, concise, objective episodic memory sentence in English.\n\n"
        "--- CURRENT TASK ---\n"
        f"Raw Input: '{prompt}'\n"
        "Clean English Episode:"
    )
    rewritten_episode, rewrite_lp_tensor = generate_with_trainable_logprobs(
        rewrite_prompt, tokenizer, model, device, max_tokens=256, temp=0.3
    )
    rewritten_episode = rewritten_episode.strip().split("\n")[0]

    if not rewritten_episode:
        rewritten_episode = "Explored the world or performed standard tasks."

    # In die DB schreiben
    clean_input = rewritten_episode.replace("'", "''")
    init_sql = f"INSERT INTO episodic_memory (content) VALUES ('{clean_input}');"
    execute_andy_sql_safely(conn, init_sql)

    # Schritt 2: Reranking
    existing_memories = query_database_memories(conn)
    retrieved_context = get_top_k_memories(rewritten_episode, existing_memories, ce_model, ce_tokenizer, device,
                                           top_k=5)

    # Schritt 3: Micro-Ebene (SQL)
    few_shot_prompt = (
        "You are a Minecraft bot named Andy with a relational SQL database as your brain.\n"
        "Analyze the situation, write a short response, and output a valid SQL command.\n"
        "CRITICAL RULE: Only use UPDATE or DELETE if a specific ID from the memories matches your current situation. Otherwise, do not perform additional database actions.\n\n"
        "--- CURRENT SITUATION ---\n"
        "Relevant memories from your SQL database:\n"
        f"{retrieved_context if retrieved_context else '[No memories found]'}\n\n"
        "Current situation / New Episode:\n"
        f"{rewritten_episode}\n"
        "RESPONSE:\n"
        "EXECUTE_SQL:"
    )

    sql_part, cognitive_lp_tensor = generate_with_trainable_logprobs(
        few_shot_prompt, tokenizer, model, device, max_tokens=512, temp=0.15
    )
    sql_part = sql_part.split("\n")[0].strip()
    if "NONE" in sql_part or not sql_part:
        sql_part = ""

    return sql_part, rewritten_episode, rewrite_lp_tensor, cognitive_lp_tensor, retrieved_context


# =====================================================================
# 6. EVALUATION & ADVANTAGES
# =====================================================================
async def evaluate_single_scale(raw_log, episode, retrieved_context, sql_command, scale: str,
                                history: str = "") -> float:
    if scale == "macro":
        criteria = (
            "CRITERIA: Is High-level strategy and behavioral coherence in the retrieved text?\n"
            "Example: In which village does the bot live? What are the characteristics of the neighborhood (market, residential area)?"
            "The bot developed a strong focus on food production and farming activities over time. The bot lives in the base. "
            f"Retrieval:\n{retrieved_context if retrieved_context else '[None]'}\n"
        )
    elif scale == "meso":
        criteria = (
            "CRITERIA: Is the scene containing the bot described correctly—that is, neither too abstractly nor in too much detail?\n"
            "Example: Where is its workplace (workshop)? What is its job there? "
            "The bot's immediate goal is to harvest the 4 ripe wheat blocks at its base to ensure it has enough food before embarking on a mining expedition at sunrise. "
            "Is it concise, objective, free of hallucinations, and aligned with the retrieved context?"
            f"Retrieval:\n{retrieved_context if retrieved_context else '[None]'}\n"
        )
    elif scale == "micro":
        criteria = (
            "CRITERIA: Does the retrieve text contain detailed information?\n"
            "What are its typical morning routines? What is the current status of its tools? "
            "The bot has identified that it can use its stone pickaxe to mine coal ore, which is essential for smelting raw iron into iron ingots,"
            f"Retrieval:\n{retrieved_context if retrieved_context else '[None]'}\n"
        )
    else:
        raise ValueError("Unknown scale type")

    prompt = f"""You are an expert AI quality inspector analyzing the retrieval of a RAG. 
Evaluate the following operational step based on the specific criteria provided below.
---
{criteria}
---
Task: Does the agent satisfy this specific criteria perfectly? Respond with EXACTLY 'Y' or 'N'.
Result:"""

    try:
        response = await client.chat.completions.create(
            model=TEACHER_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=5
        )
        choice = response.choices[0]
        probs = {"Y": 0.0, "N": 0.0}
        if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
            for top_logprob in choice.logprobs.content[0].top_logprobs:
                token_str = top_logprob.token.strip().upper()
                if token_str in probs:
                    probs[token_str] = math.exp(top_logprob.logprob)
        total_mass = sum(probs.values())
        return probs["Y"] / total_mass if total_mass > 0 else (1.0 if "Y" in choice.message.content.upper() else 0.0)
    except Exception as e:
        print(f"⚠️ Teacher-API Fehler [{scale}]: {e}")
        return 0.5


async def get_vector_teacher_evaluation(raw_log, episode, retrieved_context, sql_command, history=""):
    macro_task = evaluate_single_scale(raw_log, episode, retrieved_context, sql_command, "macro", history)
    meso_task = evaluate_single_scale(raw_log, episode, retrieved_context, sql_command, "meso", history)
    micro_task = evaluate_single_scale(raw_log, episode, retrieved_context, sql_command, "micro", history)
    return await asyncio.gather(macro_task, meso_task, micro_task)


def calculate_discounted_advantages(buffer, window_size=5, gamma=0.9):
    N = len(buffer)
    advantages = []
    for t in range(N):
        g_macro, g_meso, g_micro = 0.0, 0.0, 0.0
        for k in range(window_size):
            if t + k < N:
                s_mac, s_mes, s_mic = buffer[t + k]["score_vector"]
                factor = gamma ** k
                g_macro += factor * s_mac
                g_meso += factor * s_mes
                g_micro += factor * s_mic
        # Hierarchical Credit Assignment
        advantages.append((g_meso * g_macro, g_micro * g_macro))
    return advantages


# =====================================================================
# 7. MAIN TRAINING PIPELINE
# =====================================================================
if __name__ == "__main__":
    device = torch.device(DEVICE_TYPE)
    print(f"=== Starte Holist 2.0 RL Training-Pipeline ===")

    db_conn = init_database()
    reset_database(db_conn)  # State Reset vor Trainingsbeginn

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, fix_mistral_regex=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Von bfloat16 auf float16 wechseln
        bnb_4bit_use_double_quant=True,
    )

    andy_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device}  # Zwingt ALLES auf deinen CUDA-Index (z.B. "cuda:0")
    )

    # 2. DIESE ZEILE HINZUFÜGEN (wichtig für LayerNorm & Stabilität)
    andy_model = prepare_model_for_kbit_training(andy_model)

    # LoRA Konfiguration mit expliziten Ziel-Modulen
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # Diese Liste ist für Qwen-Modelle korrekt:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # 3. Danach erst LoRA anwenden
    andy_model = get_peft_model(andy_model, peft_config)

    # Jetzt erst den Wrapper erstellen
    andy_model = get_peft_model(andy_model, peft_config)
    andy_model.print_trainable_parameters()

    andy_model.gradient_checkpointing_enable()

    # WICHTIG: Das Modell muss für das Erzeugen von Gradienten im .train() Zustand sein!
    andy_model.train()
    optimizer = AdamW(andy_model.parameters(), lr=LEARNING_RATE)

    ce_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    ce_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME).to(device).eval()

    df = pd.read_parquet(PARQUET_DATA_PATH)

    trajectory_buffer = []
    recent_episodes_buffer = []
    gamma = 0.9

    print(chalk.green("\nStep 1: Sammle Trajektorien und berechne differenzierbare Graphen..."))

    # ... [Init-Code bleibt gleich]

    print(chalk.green("\nStarte Online-Training (Iteratives Update)..."))

    for i, row in df.iterrows():
        if i == 100: break

        new_memory, user_prompt = extract_summary_and_prompt(row["conversations"])
        if not user_prompt:
            continue

        history_context = ""
        if recent_episodes_buffer:
            history_context = "\n".join([f"Step -{len(recent_episodes_buffer) - idx}: {ep}" for idx, ep in enumerate(recent_episodes_buffer)])

        # 1. Trajektorie sammeln (wie gehabt)
        target_sql, rewritten_ep, rewrite_lp, cog_lp, retrieved_context = run_cognitive_loop(
            user_prompt, db_conn, ce_model, ce_tokenizer, tokenizer, andy_model, device
        )

        if target_sql:
            execute_andy_sql_safely(db_conn, target_sql)

        print(f"Retrieved context: {retrieved_context}")

        # 2. Teacher Evaluation (für das sofortige Signal)
        score_vector = asyncio.run(get_vector_teacher_evaluation(
            raw_log=user_prompt, episode=rewritten_ep,
            retrieved_context=retrieved_context, sql_command=target_sql, history=history_context
        ))

        # 3. SOFORTIGES UPDATE
        # Wir berechnen das Advantage für DIESEN EINZELNEN SCHRITT
        s_macro, s_meso, s_micro = score_vector

        print(f"Scores: macro = {s_macro} | meso = {s_meso} | micro = {s_micro}")

        REWARD_SCALE = 200.0

        scores = [s_macro, s_meso, s_micro]

        errors = [1.0 - x for x in scores]

        # Hier nutzen wir das Advantage direkt aus dem Score (ohne 5-Schritt-Warten)
        step_loss = rewrite_lp * cog_lp * statistics.geometric_mean(errors) * REWARD_SCALE

        # Gradienten anwenden
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            step_loss.backward()
        print(f"📉 Berechneter Batch-Loss: {step_loss.item():.4f}")
        torch.nn.utils.clip_grad_norm_(andy_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging des Fortschritts
        print(f"Step {i + 1} | Loss: {step_loss.item():.4f} | Macro: {s_macro:.2f}")

        # ... nach optimizer.step() ...
        optimizer.zero_grad(set_to_none=True)  # Wichtig: set_to_none gibt Speicher frei statt nur auf 0 zu setzen
        del step_loss
        import gc

        gc.collect()
        torch.cuda.empty_cache()  # Bereinigt den Fragmentierungs-Puffer



    # Speicher leeren und DB für den nächsten Run vorbereiten
    db_conn.close()
    print("\n Run erfolgreich beendet.")
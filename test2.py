import os
import math
import re
import gc
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# PFADE
LOCAL_MODEL_PATH = "/home/benito/PycharmProjects/Holist 2.0/models/Mindcraft-CE/Andy-4.2-Micro"
CROSS_ENCODER_WEIGHTS = "gated_memory_cross_encoder.pth"
PARQUET_DATA_PATH = "andy-4.1.parquet"

# KONFIGURATION
MAX_SEQ_LEN = 128
MAX_SAMPLES = 80
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# ARCHITEKTUR (Jetzt EXAKT identisch zum erfolgreichen Training!)
# =====================================================================
class GatedMemoryCrossEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_entries = 128  # Die feste Obergrenze

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.global_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.memory_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, query_vec: torch.Tensor, db_vectors: torch.Tensor, current_idx: int = None):
        query_vec = query_vec.float()
        db_vectors = db_vectors.float()

        # 1. Begrenzung auf das 128er-Fenster (Sliding Window für die DB)
        if db_vectors.size(0) > self.max_entries:
            shift = db_vectors.size(0) - self.max_entries
            db_vectors = db_vectors[-self.max_entries:]
            if current_idx is not None:
                current_idx = max(0, current_idx - shift)

        N = db_vectors.size(0)
        device = db_vectors.device

        # 2. Padding-Matrix erstellen
        padded_db = torch.zeros(self.max_entries, self.embed_dim, device=device, dtype=torch.float32)
        padded_db[:N] = db_vectors

        # 3. Dynamische Key-Padding-Maske erstellen
        attention_mask = torch.zeros(1, self.max_entries, dtype=torch.bool, device=device)

        if current_idx is not None:
            visible_entries = min(N, current_idx + 1)
        else:
            visible_entries = N

        # Maskiere die Zukunft UND das restliche 128er-Padding
        attention_mask[0, visible_entries:] = True

        # 4. Bi-Encoder Scoring
        q_proj = self.query_proj(query_vec)
        k_proj = self.key_proj(padded_db)
        bi_scores = torch.matmul(q_proj, k_proj.t()).squeeze(0)

        if visible_entries < self.max_entries:
            bi_scores[visible_entries:] = -float('inf')

        bi_weights = torch.softmax(bi_scores * 5.0, dim=-1)

        # 5. Cross-Attention
        q = query_vec.unsqueeze(1)
        k = padded_db.unsqueeze(0)
        v = padded_db.unsqueeze(0)

        cross_output, _ = self.global_attention(
            query=q, key=k, value=v,
            key_padding_mask=attention_mask
        )
        cross_output = cross_output.squeeze(1)

        # 6. Fusions- & Gating-Netzwerk
        merged = torch.cat([query_vec, cross_output], dim=-1)
        cross_encoded_vector = self.memory_fusion(merged)

        final_output_vector = query_vec + self.beta * cross_encoded_vector

        return final_output_vector, bi_weights


def get_embedding(text: str, tokenizer, model, device, layer_idx: int = None) -> torch.Tensor:
    if layer_idx is None:
        layer_idx = model.config.num_hidden_layers // 2

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx + 1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    res = embedding.squeeze(0).float()
    del inputs, outputs, hidden_states, attention_mask
    return res


def get_decoder_layers(causal_model):
    candidates = [("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")]
    for outer, inner in candidates:
        if hasattr(causal_model, outer) and hasattr(getattr(causal_model, outer), inner):
            return getattr(getattr(causal_model, outer), inner)
    raise AttributeError("Konnte Decoder-Layer nicht finden.")


def build_prefix_text(memory_text: str) -> str:
    return f"Du bist Andy, ein Minecraft-Bot.\nSummarized memory: '{memory_text}'\n"


def build_suffix_text(prompt: str) -> str:
    return f"User: {prompt}\nAndy:"


# =====================================================================
# INFERENZ-PIPELINE
# =====================================================================
def run_interaction_pipeline(prompt: str, db_vectors: torch.Tensor, cross_encoder, tokenizer,
                             causal_model, device, layer_idx: int, target_idx: int, injection_scale: float = 1.0):
    cross_encoder.eval()

    query_vec = get_embedding(prompt, tokenizer, causal_model, device, layer_idx=layer_idx).unsqueeze(0)
    with torch.no_grad():
        # NEU: Reiche target_idx als current_idx weiter, damit die Inferenz die Timeline respektiert
        final_output_vector, _ = cross_encoder(query_vec, db_vectors, current_idx=target_idx)

    suffix_text = build_suffix_text(prompt)
    student_prefix = build_prefix_text("")
    full_text = student_prefix + suffix_text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    layers = get_decoder_layers(causal_model)
    target_layer = layers[layer_idx]
    model_dtype = next(causal_model.parameters()).dtype
    inj_vec = final_output_vector.reshape(1, 1, -1).to(model_dtype)

    def injection_hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        target_norm = hidden_states.norm(dim=-1).mean().detach()
        current_norm = inj_vec.norm(dim=-1).mean().detach().clamp(min=1e-6)
        norm_scale = (target_norm / current_norm) * 0.1
        injected = hidden_states + injection_scale * norm_scale * inj_vec
        return (injected,) + output[1:] if isinstance(output, tuple) else injected

    handle = target_layer.register_forward_hook(injection_hook)

    try:
        with torch.no_grad():
            output_ids = causal_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    finally:
        handle.remove()

    return generated_text


# =====================================================================
# METRIK
# =====================================================================
def bm25_grounding_check(generated_text: str, database: list, target_episode_idx: int):
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    corpus = [tokenize(doc["text"]) for doc in database]
    target_tokens = corpus[target_episode_idx]
    gen_tokens = tokenize(generated_text)

    df = Counter()
    for doc in corpus:
        df.update(set(doc))

    N = len(corpus)
    avgdl = sum(len(doc) for doc in corpus) / (N if N > 0 else 1)
    k1, b = 1.5, 0.75

    bm25_target_score = 0.0
    gen_counter = Counter(gen_tokens)
    doc_len = len(gen_tokens)

    for word in set(target_tokens):
        if word in gen_counter:
            word_df = df[word]
            idf = math.log((N - word_df + 0.5) / (word_df + 0.5) + 1.0)
            tf = gen_counter[word]
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            bm25_target_score += score

    return None, None, bm25_target_score


# =====================================================================
# DATENVERARBEITUNG
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


def load_and_process_data(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    processed_data = []
    for _, row in df.iterrows():
        summary, prompt = extract_summary_and_prompt(row["conversations"])
        if summary and prompt:
            processed_data.append({"summary": summary, "prompt": prompt})
        if MAX_SAMPLES and len(processed_data) >= MAX_SAMPLES:
            break
    return processed_data


# =====================================================================
# MAIN EVALUATION
# =====================================================================
if __name__ == "__main__":
    device = torch.device(DEVICE_TYPE)
    print(f"Lade Evaluierungsumgebung auf: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, fix_mistral_regex=True)
    # Nutze float16 passend zum Hook
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    LAYER_IDX = model.config.num_hidden_layers // 2
    embed_dim = model.config.hidden_size

    cross_encoder = GatedMemoryCrossEncoder(embed_dim=embed_dim).to(device)

    if os.path.exists(CROSS_ENCODER_WEIGHTS):
        print(f"Lade trainierte Cross-Encoder Gewichte aus '{CROSS_ENCODER_WEIGHTS}'...")
        cross_encoder.load_state_dict(torch.load(CROSS_ENCODER_WEIGHTS, map_location=device))
        cross_encoder.eval()
    else:
        raise FileNotFoundError(
            f"Die Gewichtsdatei {CROSS_ENCODER_WEIGHTS} wurde nicht gefunden!")

    training_examples = load_and_process_data(PARQUET_DATA_PATH)
    all_summaries = [ex["summary"] for ex in training_examples]
    database_for_teacher = [{"text": s} for s in all_summaries]

    print("Berechne Episoden-Embeddings für den Suchraum...")
    db_vectors_list = []
    for i, s in enumerate(all_summaries):
        db_vectors_list.append(get_embedding(s, tokenizer, model, device, layer_idx=LAYER_IDX).unsqueeze(0))
    db_vectors = torch.cat(db_vectors_list, dim=0).to(device)
    del db_vectors_list
    gc.collect()

    # =====================================================================
    # INTERAKTIVE TESTSCHLEIFE
    # =====================================================================
    print("\n" + "=" * 60)
    print(" READY FOR TESTING: Du kannst jetzt Test-Indizes prüfen.")
    print(" =" * 60)

    while True:
        eingabe = input(
            f"\nGib einen Index (0 bis {len(training_examples) - 1}) zum Testen ein (oder 'q' zum Beenden): ")
        if eingabe.lower() == 'q':
            break

        try:
            test_idx = int(eingabe)
            if not (0 <= test_idx < len(training_examples)):
                print(f"Bitte einen Index zwischen 0 und {len(training_examples) - 1} wählen.")
                continue
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl eingeben.")
            continue

        test_beispiel = training_examples[test_idx]
        test_frage = test_beispiel["prompt"]

        print(f"\n[Test-Prompt]: '{test_frage}'")
        print(f"[Zielepisode]: '{database_for_teacher[test_idx]['text']}'")

        # 4. Antwort generieren
        antwort = run_interaction_pipeline(
            prompt=test_frage,
            db_vectors=db_vectors,
            cross_encoder=cross_encoder,
            tokenizer=tokenizer,
            causal_model=model,
            device=device,
            layer_idx=LAYER_IDX,
            target_idx=test_idx
        )

        # 5. Aufmerksamkeit des Cross-Encoders auslesen
        with torch.no_grad():
            query_vec = get_embedding(test_frage, tokenizer, model, device, layer_idx=LAYER_IDX).unsqueeze(0)
            _, final_attention_weights = cross_encoder(query_vec, db_vectors, current_idx=test_idx)

            top_retrieved_idx = torch.argmax(final_attention_weights).item()
            top_weight = final_attention_weights[top_retrieved_idx].item()

        # ─────────────────────────────────────────────────────────────────
        # FIX: Hier ist der fehlende Aufruf, der das "Rot" wegmacht!
        # ─────────────────────────────────────────────────────────────────
        _, _, bm25_target_score = bm25_grounding_check(
            antwort, database_for_teacher, target_episode_idx=test_idx
        )
        # ─────────────────────────────────────────────────────────────────

        print("\n--- ERGEBNISSE ---")
        # Zurückrechnen auf den echten globalen Index der DB für das Printout
        global_retrieved_idx = top_retrieved_idx if test_idx < 128 else (test_idx - 127 + top_retrieved_idx)

        print(f"Cross-Encoder Fokus auf Episode-Index: {global_retrieved_idx} (Konfidenz: {top_weight:.4f})")
        if global_retrieved_idx == test_idx:
            print("🎉 ERFOLG: Der mathematische Fokus liegt exakt auf der richtigen Episode!")
        else:
            print(f"⚠️ ABWEICHUNG: Modell fokussiert Episode {global_retrieved_idx} statt {test_idx}.")

        print(f"BM25-Score der Antwort gegen Zielepisode: {bm25_target_score:.4f}")
        print(f"\nGenerierte Antwort des Bots Andy:\n{'-' * 50}\n{antwort}\n{'-' * 50}")
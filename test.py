import os
import math
import re
import gc
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # <- Neu für 4-Bit
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

LOCAL_MODEL_PATH = "/home/benito/PycharmProjects/Holist 2.0/models/Mindcraft-CE/Andy-4.2-Micro"

# KONFIGURATION FÜR RAM-SCHONENDES TRAINING
MAX_SEQ_LEN = 128
MAX_SAMPLES = 100  # Kannst du jetzt höher schrauben, da wir VRAM sparen!


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
        self.beta = 1.0

    def forward(self, query_vec: torch.Tensor, db_vectors: torch.Tensor, current_idx: int = None):
        query_vec = query_vec.float()
        db_vectors = db_vectors.float()

        # 1. Begrenzung auf das 128er-Fenster (Sliding Window für die DB)
        if db_vectors.size(0) > self.max_entries:
            # Wenn wir abschneiden, müssen wir auch den current_idx anpassen!
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

        # ─────────────────────────────────────────────────────────────────
        # NEU: Das Kausale Maskierungs-Feature
        # ─────────────────────────────────────────────────────────────────
        if current_idx is not None:
            # Im Training: Alles, was NACH der aktuellen Episode kommt, ist Zukunft!
            # Wenn current_idx = 5 ist, darf das Modell nur 0 bis 5 sehen.
            # Ab Index 6 (also current_idx + 1) wird alles blockiert.
            visible_entries = min(N, current_idx + 1)
        else:
            # Im Live-Spiel (Inferenz): Der Bot sieht standardmäßig alle existierenden Einträge
            visible_entries = N

        # Maskiere die Zukunft UND das restliche 128er-Padding
        attention_mask[0, visible_entries:] = True
        # ─────────────────────────────────────────────────────────────────

        # 4. Bi-Encoder Scoring (für die bi_weights)
        q_proj = self.query_proj(query_vec)
        k_proj = self.key_proj(padded_db)
        bi_scores = torch.matmul(q_proj, k_proj.t()).squeeze(0)

        # Auch hier die Zukunft/Padding auf -inf setzen, damit der Softmax sie ignoriert
        if visible_entries < self.max_entries:
            bi_scores[visible_entries:] = -float('inf')

        bi_weights = torch.softmax(bi_scores * 5.0, dim=-1)

        # 5. Cross-Attention (Nutzt jetzt die kausale Maske)
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

        # FIX: Kein hartes 'query_vec +' mehr!
        # Wenn beta = 0 ist, ist der finale Vektor jetzt absolut 0.
        final_output_vector = self.beta * cross_encoded_vector

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

    res = embedding.squeeze(0).float()  # Zurück auf float32 für den Cross-Encoder
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
# TEACHER-STUDENT-DISTILLATION
# =====================================================================
def train_distillation_step(cross_encoder, causal_model, tokenizer, optimizer, database, db_vectors,
                            example: dict, device, layer_idx: int, injection_scale: float = 1.0,
                            retrieval_loss_weight: float = 0.3):
    cross_encoder.train()
    optimizer.zero_grad()

    # WICHTIG: db_vectors liegt auf der CPU. Wir kopieren es nur für diesen Schritt auf die GPU
    db_vectors_gpu = db_vectors.to(device)

    # 1. Query-Embedding
    query_vec = get_embedding(example["prompt"], tokenizer, causal_model, device, layer_idx=layer_idx).unsqueeze(0)

    current_idx = example["target_episode_idx"]  # Das entspricht dem "i" aus deiner Schleife

    # ÄNDERUNG: current_idx wird jetzt explizit an den Cross-Encoder übergeben!
    final_output_vector, final_attention_weights = cross_encoder(query_vec, db_vectors_gpu, current_idx=current_idx)

    # ÄNDERUNG: Mathematischer Index-Fix für das 128er Sliding-Window
    target_idx = current_idx if current_idx < 128 else 127
    retrieval_loss = -torch.log(final_attention_weights[target_idx] + 1e-8)
    # ─────────────────────────────────────────────────────────────────

    # 2. Suffix-Tokenisierung
    suffix_text = build_suffix_text(example["prompt"])
    suffix_inputs = tokenizer(suffix_text, return_tensors="pt", padding=True, truncation=True,
                              max_length=MAX_SEQ_LEN).to(device)
    suffix_len = suffix_inputs.input_ids.shape[1]

    # 3. TEACHER-Pass (Numerisch stabilisiert im Log-Raum)
    teacher_prefix = build_prefix_text(database[example["target_episode_idx"]]["text"])
    teacher_full_text = teacher_prefix + suffix_text
    teacher_inputs = tokenizer(teacher_full_text, return_tensors="pt", padding=True, truncation=True,
                               max_length=MAX_SEQ_LEN).to(device)

    with torch.no_grad():
        teacher_outputs = causal_model(**teacher_inputs)
        # WICHTIG: log_softmax statt softmax! Das verhindert die 0.0-Unterläufe.
        teacher_log_probs = torch.log_softmax(teacher_outputs.logits[:, -suffix_len:, :].float(), dim=-1).detach()

    del teacher_inputs, teacher_outputs
    gc.collect()

    # 4. STUDENT-Pass
    student_prefix = build_prefix_text("")
    student_full_text = student_prefix + suffix_text
    student_inputs = tokenizer(student_full_text, return_tensors="pt", padding=True, truncation=True,
                               max_length=MAX_SEQ_LEN).to(device)

    layers = get_decoder_layers(causal_model)
    target_layer = layers[layer_idx]

    # FIX: Wir nutzen bfloat16, weil das Modell kein float16 verwendet!
    model_dtype = torch.bfloat16
    inj_vec = final_output_vector.reshape(1, 1, -1).to(device).to(model_dtype)

    def injection_hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        target_norm = hidden_states.norm(dim=-1).mean().detach()
        current_norm = inj_vec.norm(dim=-1).mean().detach().clamp(min=1e-6)
        norm_scale = (target_norm / current_norm) * 0.1

        delta = (injection_scale * norm_scale * inj_vec).to(hidden_states.dtype)
        injected = hidden_states + delta

        return (injected,) + output[1:] if isinstance(output, tuple) else injected

    handle = target_layer.register_forward_hook(injection_hook)

    try:
        student_outputs = causal_model(**student_inputs)
        student_log_probs = torch.log_softmax(student_outputs.logits[:, -suffix_len:, :].float(), dim=-1)
    finally:
        handle.remove()

    # 5. KL-Divergenz Loss (Mit log_target=True absolut immun gegen NaN durch 0.0)
    kl_loss = nn.functional.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True
    ).sum(dim=-1).mean()

    # Gesamt-Loss
    total_loss = kl_loss + retrieval_loss_weight * retrieval_loss
    total_loss.backward()

    # GRADIENT CLIPPING: Verhindert, dass die Gewichte bei float16 explodieren
    torch.nn.utils.clip_grad_norm_(cross_encoder.parameters(), max_norm=1.0)

    optimizer.step()

    res = (total_loss.item(), kl_loss.item(), retrieval_loss.item())

    # Aggressives VRAM Aufräumen am Ende des Schritts
    del student_inputs, student_outputs, student_log_probs, total_loss, db_vectors_gpu, query_vec
    torch.cuda.empty_cache()
    gc.collect()

    return res


# =====================================================================
# DATENLADE-LOGIK
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
            #print(f"summary: {summary} \n\n prompt: {prompt}")
        if MAX_SAMPLES and len(processed_data) >= MAX_SAMPLES:
            break
    return processed_data


if __name__ == "__main__":
    # 1. Device auf CUDA umstellen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte optimiertes Training auf: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, fix_mistral_regex=True)
    tokenizer.truncation_side = "left"

    # 2. Modell nativ in bfloat16 direkt auf die GPU laden
    print("Lade Hauptmodell auf die GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # 3. WICHTIG: Gewichte des Causal-Modells einfrieren, um VRAM bei .backward() zu sparen
    for param in model.parameters():
        param.requires_grad = False

    # 4. DYNAMISCHER FIX: Alle Conv1d-Schichten an Device & Dtype anpassen
    # (Hugging Face vergisst diese Schichten bei Qwen-Modellen manchmal im Float32-Modus)
    target_dtype = next(model.parameters()).dtype
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            module.to(device=device, dtype=target_dtype)

    embed_dim = model.config.hidden_size
    # Cross-Encoder ebenfalls auf die GPU schieben
    cross_encoder = GatedMemoryCrossEncoder(embed_dim=embed_dim).to(device)
    optimizer = optim.AdamW(cross_encoder.parameters(), lr=5e-4)

    training_examples = load_and_process_data("andy-4.1.parquet")
    print(f"Geladen: {len(training_examples)} Beispiele.")

    all_summaries = [ex["summary"] for ex in training_examples]
    database_for_teacher = [{"text": s} for s in all_summaries]

    # 5. Embeddings berechnen (läuft jetzt blitzschnell auf CUDA)
    print("Berechne Embeddings für die Datenbank...")
    db_vectors_list = []
    for i, s in enumerate(all_summaries):
        db_vectors_list.append(get_embedding(s, tokenizer, model, device).unsqueeze(0))
        if (i + 1) % 50 == 0:
            print(f"  Fortschritt: {i + 1}/{len(all_summaries)}")

    db_vectors = torch.cat(db_vectors_list, dim=0)
    del db_vectors_list
    gc.collect()

    print("Starte Training...")
    for epoch in range(15):
        for i, ex in enumerate(training_examples):
            example_for_step = {"prompt": ex["prompt"], "target_episode_idx": i}
            loss, kl, ret = train_distillation_step(
                cross_encoder, model, tokenizer, optimizer, database_for_teacher, db_vectors,
                example_for_step, device, model.config.num_hidden_layers // 2
            )
            print(f"Epoch {epoch + 1}, Schritt {i + 1}: Loss {loss:.4f} (KL {kl:.4f}, Ret {ret:.4f})")

    torch.save(cross_encoder.state_dict(), "gated_memory_cross_encoder.pth")
    print("Fertig. Modell gespeichert.")
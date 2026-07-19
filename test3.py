import os
import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL_PATH = "/home/benito/PycharmProjects/Holist 2.0/models/Mindcraft-CE/Andy-4.2-Micro"
CROSS_ENCODER_WEIGHTS = "gated_memory_cross_encoder.pth"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


class GatedMemoryCrossEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_entries = 128
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

        N = db_vectors.size(0)
        device = db_vectors.device

        padded_db = torch.zeros(self.max_entries, self.embed_dim, device=device, dtype=torch.float32)
        padded_db[:N] = db_vectors

        attention_mask = torch.zeros(1, self.max_entries, dtype=torch.bool, device=device)
        visible_entries = min(N, current_idx + 1) if current_idx is not None else N
        attention_mask[0, visible_entries:] = True

        q_proj = self.query_proj(query_vec)
        k_proj = self.key_proj(padded_db)
        bi_scores = torch.matmul(q_proj, k_proj.t()).squeeze(0)

        if visible_entries < self.max_entries:
            bi_scores[visible_entries:] = -float('inf')

        bi_weights = torch.softmax(bi_scores * 5.0, dim=-1)

        q = query_vec.unsqueeze(1)
        k = padded_db.unsqueeze(0)
        v = padded_db.unsqueeze(0)

        cross_output, _ = self.global_attention(query=q, key=k, value=v, key_padding_mask=attention_mask)
        cross_output = cross_output.squeeze(1)

        merged = torch.cat([query_vec, cross_output], dim=-1)
        cross_encoded_vector = self.memory_fusion(merged)
        final_output_vector = self.beta * cross_encoded_vector

        return final_output_vector, bi_weights


def get_embedding(text: str, tokenizer, model, device, layer_idx: int) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx + 1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    res = embedding.squeeze(0).float()
    del inputs, outputs, hidden_states, attention_mask
    return res


def get_decoder_layers(causal_model):
    for outer, inner in [("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")]:
        if hasattr(causal_model, outer) and hasattr(getattr(causal_model, outer), inner):
            return getattr(getattr(causal_model, outer), inner)
    raise AttributeError("Konnte Decoder-Layer nicht finden.")


def run_isolated_test(prompt: str, db_vectors: torch.Tensor, cross_encoder, tokenizer,
                      causal_model, device, layer_idx: int, injection_scale: float):
    cross_encoder.eval()
    query_vec = get_embedding(prompt, tokenizer, causal_model, device, layer_idx=layer_idx).unsqueeze(0)

    with torch.no_grad():
        final_output_vector, bi_weights = cross_encoder(query_vec, db_vectors, current_idx=0)

    full_text = "Du bist Andy, ein Minecraft-Bot.\nSummarized memory: ''\nUser: " + prompt + "\nAndy:"
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
                max_new_tokens=40,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["User:", "System output:", "<think>"],
                tokenizer=tokenizer
            )
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    finally:
        handle.remove()

    return generated_text, bi_weights[0].item(), cross_encoder.beta


# =====================================================================
# EXECUTION (MULTI-EXAMPLE EVALUATION)
# =====================================================================
if __name__ == "__main__":
    device = torch.device(DEVICE_TYPE)
    print(f"Lade isolierte Testumgebung auf: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    LAYER_IDX = model.config.num_hidden_layers // 2
    embed_dim = model.config.hidden_size

    cross_encoder = GatedMemoryCrossEncoder(embed_dim=embed_dim).to(device)
    if os.path.exists(CROSS_ENCODER_WEIGHTS):
        cross_encoder.load_state_dict(torch.load(CROSS_ENCODER_WEIGHTS, map_location=device))
        cross_encoder.eval()
        print("-> Trainierte Cross-Encoder-Gewichte erfolgreich geladen.\n")
    else:
        print("⚠️ Warnung: Keine trainierten Gewichte gefunden, nutze untrainierte Initialisierung.\n")

    # ─────────────────────────────────────────────────────────────────
    # DEIN EVALUATIONS-SET (VERSCHIEDENE SZENARIEN)
    # ─────────────────────────────────────────────────────────────────
    test_cases = [
        {
            "id": "1. Wirtschaft & Farming (Dein Original)",
            "memory": "This morning, the bot decided to go to the mine to mine iron ore, but it didn't have enough bread in its inventory; so, it first went to the field to harvest 6 units of grain, from which it crafted 2 loaves of bread.",
            "prompt": "What have you done this morning?"
        },
        {
            "id": "2. Kampf & Basis-Verteidigung",
            "memory": "Yesterday evening, a creeper surprised the bot near the storage room and blew up two chests. The bot spent the whole night rebuilding the stone wall and sorting the spilled items back into new chests.",
            "prompt": "Did anything happen to the base recently?"
        },
        {
            "id": "3. Progression & Mining",
            "memory": "An hour ago, the bot finally found 5 diamonds at Y-layer -58. It immediately returned to the base and crafted a diamond pickaxe to replace its broken iron tool.",
            "prompt": "Do you have any new tools?"
        }
    ]

    # Schleife über alle Testfälle
    for case in test_cases:
        print("=" * 70)
        print(f"SZENARIO: {case['id']}")
        print(f"[Gedächtnis]: '{case['memory']}'")
        print(f"[Prompt]    : '{case['prompt']}'")
        print("=" * 70)

        # Berechne den spezifischen DB-Vektor für dieses Gedächtnis
        db_vector = get_embedding(case['memory'], tokenizer, model, device, layer_idx=LAYER_IDX).unsqueeze(0)

        # Teste jede Injektionsstärke für dieses Szenario
        for scale in [0.0, 0.5, 1.5]:
            antwort, att_weight, beta_val = run_isolated_test(
                case['prompt'], db_vector, cross_encoder, tokenizer, model, device, LAYER_IDX, injection_scale=scale
            )

            print(f"--- Scale = {scale} (Beta: {beta_val:.4f}) | Attention: {att_weight * 100:.1f}% ---")
            print(f"Andy: {antwort.strip()}\n")
        print("\n")
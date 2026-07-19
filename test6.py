import asyncio
import math
import random
from openai import AsyncOpenAI, RateLimitError

# Initialisierung des asynchronen Clients
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"


async def evaluate_pair_direction(cand_1, cand_2, task_description):
    """Führt eine einzelne Inferenz für ein Paar in einer spezifischen Reihenfolge aus."""
    prompt = f"""Task to evaluate: {task_description}

Candidate A:
{cand_1}

Candidate B:
{cand_2}

Which candidate provided the better, more accurate, and more robust solution? 
Analyze their trajectory internally, then respond with EXACTLY 'A' if Candidate A is better, or 'B' if Candidate B is better. Do not output any other text.

Result:"""

    max_retries = 5
    delay = 1.0
    for retries in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
                extra_body={
                    "provider": {
                        "allow_fallbacks": True,
                        "require_parameters": True,
                        "order": ["DeepInfra", "Amazon Bedrock"]
                    }
                }
            )
            break
        except RateLimitError as e:
            if retries == max_retries - 1:
                raise e
            await asyncio.sleep(delay)
            delay *= 2
        except Exception:
            return {"A": 0.5, "B": 0.5}

    choice = response.choices[0]
    probs = {"A": 0.0, "B": 0.0}

    if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
        token_logprobs_data = choice.logprobs.content[0].top_logprobs
        for top_logprob in token_logprobs_data:
            token_str = top_logprob.token.strip()
            if token_str in probs:
                probs[token_str] = math.exp(top_logprob.logprob)

    total_mass = sum(probs.values())
    if total_mass == 0:
        # Fallback auf Hard-Label falls Logprobs außerhalb Top-5
        text_resp = choice.message.content.strip() if choice.message.content else ""
        return {"A": 1.0 if text_resp == "A" else 0.0, "B": 1.0 if text_resp == "B" else 0.0}

    return {k: v / total_mass for k, v in probs.items()}


async def get_pairwise_score(cand_a, cand_b, task_description):
    """Berechnet den robusteren, positionsbereinigten Score für ein Kandidatenpaar."""
    # Starte beide Richtungen parallel, um Latenz zu minimieren
    fwd_task = evaluate_pair_direction(cand_a, cand_b, task_description)
    bwd_task = evaluate_pair_direction(cand_b, cand_a, task_description)

    fwd_probs, bwd_probs = await asyncio.gather(fwd_task, bwd_task)

    # Formel: (P(A > B) + (1 - P(B > A))) / 2
    score_a = (fwd_probs["A"] + (1.0 - bwd_probs["B"])) / 2.0
    return score_a


async def run_probabilistic_pivot_tournament(candidates, task_description, k=3):
    """
    Sortiert Kandidaten mittels Probabilistic Pivot Tournament (PPT).
    Reduziert die Komplexität von O(N^2) auf O(N * k).
    """
    if len(candidates) <= 1:
        return candidates

    # 1. Pivot-Auswahl: Bestimme den besten Kandidaten aus einem zufälligen k-Subset
    sample_indices = random.sample(range(len(candidates)), min(k, len(candidates)))
    pivot_idx = sample_indices[0]

    for idx in sample_indices[1:]:
        score = await get_pairwise_score(candidates[idx], candidates[pivot_idx], task_description)
        if score > 0.5:
            pivot_idx = idx

    pivot_candidate = candidates[pivot_idx]

    # 2. Partitionierung: Vergleiche alle verbleibenden Elemente asynchron gegen den Pivot
    left_side = []  # Besser als Pivot
    right_side = []  # Schlechter als Pivot

    tasks = []
    remaining_candidates = [c for i, c in enumerate(candidates) if i != pivot_idx]

    for cand in remaining_candidates:
        tasks.append(get_pairwise_score(cand, pivot_candidate, task_description))

    scores = await asyncio.gather(*tasks)

    for cand, score in zip(remaining_candidates, scores):
        if score > 0.5:
            left_side.append(cand)
        else:
            right_side.append(cand)

    # 3. Rekursiver Merge Schritt
    sorted_left = await run_probabilistic_pivot_tournament(left_side, task_description, k)
    sorted_right = await run_probabilistic_pivot_tournament(right_side, task_description, k)

    return sorted_left + [pivot_candidate] + sorted_right


# --- Test Execution ---
async def main():
    task = "Write a python function to compute the Fibonacci sequence efficiently."

    outputs = [
        "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",  # Schlechte Rekursion
        "def fib(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        # Optimale Iteration
        "def fib(n):\n    memo = {0:0, 1:1}\n    if n not in memo:\n        memo[n] = fib(n-1) + fib(n-2)\n    return memo[n]"
        # Fehlerhafter Cache-Scope
    ]

    print("Starte Probabilistic Pivot Tournament...")
    sorted_candidates = await run_probabilistic_pivot_tournament(outputs, task, k=2)

    print("\n--- RANKING ERGEBNIS (Beste zuerst) ---")
    for i, cand in enumerate(sorted_candidates, 1):
        print(f"Platz {i}:\n{cand}\n" + "-" * 20)


if __name__ == "__main__":
    asyncio.run(main())
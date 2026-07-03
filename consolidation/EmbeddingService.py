import multiprocessing
import queue
from consolidation.Andy import EmbeddingModel  # Pfad zu deiner Andy.py anpassen


def central_embedding_worker(request_queue, response_queues):
    """
    Läuft in einem EINZIGEN Prozess und hält das Modell auf der GPU.
    """
    print("🧠 [Embedding-Service] Lade Modell auf die GPU...")
    embedder = EmbeddingModel(
        model_type="andy",
        model_name="/home/benito/PycharmProjects/Holist 2.0/models/Mindcraft-CE/Andy-4.2-Micro",
        device="cuda"  # Nur hier wird CUDA blockiert!
    )
    print("✅ [Embedding-Service] Modell erfolgreich auf GPU geladen. Bereit für Anfragen.")

    while True:
        try:
            # Erwartet ein Tuple: (bot_name, text_to_embed)
            bot_name, text = request_queue.get()

            if bot_name == "SHUTDOWN":
                break

            # Vektor generieren
            vector = embedder.create(text)

            # Ergebnis zurück in die spezifische Queue des Bots legen
            if bot_name in response_queues:
                response_queues[bot_name].put(vector)

        except Exception as e:
            print(f"❌ Fehler im Embedding-Service: {e}")
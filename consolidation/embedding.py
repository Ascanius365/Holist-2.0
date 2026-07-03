import os
import numpy as np
from tqdm import tqdm
from typing import List, Union, Optional, Literal
import torch
from dotenv import load_dotenv

# OpenAI API
from openai import OpenAI

# Sentence Transformers
from sentence_transformers import SentenceTransformer

load_dotenv()


class EmbeddingModel:
    """
    Unified embedding model class that supports OpenAI, OpenRouter, and SentenceTransformer models
    """

    def __init__(
        self,
        model_type: Literal["openai", "sentence_transformer", "openrouter"],
        model_name: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the embedding model

        Args:
            model_type: Type of model to use ("openai", "sentence_transformer", or "openrouter")
            model_name: Name of the model to use
                        - For OpenAI: "text-embedding-3-large", etc.
                        - For OpenRouter: e.g., "intfloat/e5-large-v2"
                        - For SentenceTransformer: "multi-qa-mpnet-base-dot-v1", etc.
            api_key: API key for OpenAI or OpenRouter (falls nicht über Umgebungsvariablen gesetzt)
            base_url: Optionale benutzerdefinierte Basis-URL für die API
            device: Device to use for computation ("cpu" or "cuda")
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device

        if self.model_type == "sentence_transformer":
            self._model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=True
            )
        elif self.model_type == "openrouter":
            # OpenRouter Konfiguration abfangen
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            base_url = base_url or "https://openrouter.ai/api/v1"
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        elif self.model_type == "openai":
            # Standard OpenAI Konfiguration
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def create(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Create embeddings for the given texts

        Args:
            texts: A string or list of strings to embed
            batch_size: Batch size for processing
                        - Default for API models: 1536
                        - Default for SentenceTransformer: 32
            dimensions: Dimension of the embeddings (only used for OpenAI models if supported)
            show_progress_bar: Whether to show a progress bar
            **kwargs: Additional arguments passed to the underlying embedding method

        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Replace empty strings with space to prevent errors
        texts = [text if text != "" else " " for text in texts]

        # Set default batch size based on model type
        if batch_size is None:
            batch_size = 32 if self.model_type == "sentence_transformer" else 1536

        if self.model_type in ["openai", "openrouter"]:
            return self._create_api_embeddings(
                texts, batch_size, dimensions, show_progress_bar, **kwargs
            )
        else:  # sentence_transformer
            return self._create_st_embeddings(texts, batch_size, show_progress_bar, **kwargs)

    def _create_api_embeddings(
        self, texts, batch_size, dimensions, show_progress_bar, **kwargs
    ):
        """Create embeddings using OpenAI-compatible APIs (OpenAI & OpenRouter)"""
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):

            """
            # ── KONSOLEN-AUSGABE FÜR LOKALE MODELLE ──────────────────────────────────
            print(
                f"\n=== [LOCAL-DEBUG] Processing Batch ... with SentenceTransformer ({self.model_name}) ===")
            for i, text in enumerate(batch_texts):
                print(f"  -> Text [{i}]: {repr(text)}")
            print("========================================================================\n")
            # ─────────────────────────────────────────────────────────────────────────"""

            embedding_args = {
                "input": batch_texts,
                "model": self.model_name,
            }

            # Add dimensions if specified (wichtig: OpenRouter/E5 unterstützt dies i.d.R. nicht)
            if dimensions is not None and self.model_type == "openai":
                embedding_args["dimensions"] = dimensions

            # Add any additional kwargs
            embedding_args.update(kwargs)

            response = self._client.embeddings.create(**embedding_args)
            response = [data.embedding for data in response.data]
            output.extend(response)

        # Da OpenRouter-Modelle standardmäßig nicht immer normalisierte Vektoren liefern,
        # jagen wir sie zur Sicherheit wie bei den ST-Modellen durch die Normalisierung.
        return self.normalize_embeddings(np.array(output))

    def _create_st_embeddings(self, texts, batch_size, show_progress_bar, **kwargs):
        """Create embeddings using SentenceTransformer"""
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):

            embeddings = self._model.encode(
                batch_texts,
                show_progress_bar=False,  # We're handling our own progress bar
                convert_to_numpy=True,
                **kwargs,
            )
            embeddings = self.normalize_embeddings(embeddings)
            output.extend(embeddings)

        return np.array(output)

    def batch_generator(self, texts, batch_size, show_progress_bar=False):
        """Generate batches of texts"""
        total_batches = int(np.ceil(len(texts) / batch_size))
        if show_progress_bar:
            model_name = (
                self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            )
            batch_range = tqdm(range(total_batches), desc=f"Creating {model_name} embeddings")
        else:
            batch_range = range(total_batches)

        for batch in batch_range:
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, len(texts))
            yield texts[batch_start:batch_end]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this model"""
        if self.model_type == "sentence_transformer":
            return self._model.get_sentence_embedding_dimension()
        else:
            # Für API-Modelle geben wir None zurück, da sich die Dimensionen dynamisch aus dem ersten Vektor ableiten lassen
            return None

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize the embeddings to have unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Verhindert Division durch Null bei leeren Vektoren
        norms = np.where(norms == 0, 1.0, norms)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings
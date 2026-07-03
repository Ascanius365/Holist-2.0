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

# Transformers für das geköpfte CausalLM (Andy)
from transformers import AutoTokenizer, AutoModel

load_dotenv()


class EmbeddingModel:
    """
    Unified embedding model class that supports OpenAI, OpenRouter,
    SentenceTransformer models, and domain-specific CausalLMs (like Andy).
    """

    def __init__(
            self,
            model_type: Literal["openai", "sentence_transformer", "openrouter", "andy"],
            model_name: str = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            #device: str = "cuda" if torch.cuda.is_available() else "cpu",
            device = "cpu",
            cache_dir: Optional[str] = "./models",
    ):
        self.model_type = model_type
        self.model_name = model_name or ("Mindcraft-CE/Andy-4.2-Micro" if model_type == "andy" else None)
        self.device = device
        self.cache_dir = cache_dir

        if self.model_type == "sentence_transformer":
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
                cache_folder=self.cache_dir
            )

        elif self.model_type == "andy":
            is_local = os.path.isdir(self.model_name)

            if is_local:
                print(f"📂 Lade Minecraft-Embedding-Modell direkt aus lokalem Ordner: '{self.model_name}'")
                load_args = {"local_files_only": True}
            else:
                print(f"📦 Lade Minecraft-Embedding-Modell via HF-Hub: '{self.model_name}'")
                load_args = {"cache_dir": self.cache_dir}

            # 'fix_mistral_regex=True' verhindert die Tokenizer-Warnung
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                fix_mistral_regex=True,
                **load_args
            )

            # 'dtype' statt 'torch_dtype' verhindert die Deprecation-Warnung
            self._model = AutoModel.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                **load_args
            ).to(self.device)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        elif self.model_type == "openrouter":
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            base_url = base_url or "https://openrouter.ai/api/v1"
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        elif self.model_type == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def get_embedding_dimension(self) -> int:
        if self.model_type == "sentence_transformer":
            return self._model.get_sentence_embedding_dimension()
        elif self.model_type == "andy":
            # Weiche Landung für die verschachtelte Qwen3.5-Multimodal-Config
            if hasattr(self._model.config, "text_config"):
                return self._model.config.text_config.hidden_size
            return getattr(self._model.config, "hidden_size", None)
        else:
            return None

    def create(
            self,
            texts: Union[str, List[str]],
            batch_size: Optional[int] = None,
            dimensions: Optional[int] = None,
            show_progress_bar: bool = False,
            **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = [text if text != "" else " " for text in texts]

        if batch_size is None:
            batch_size = 32 if self.model_type in ["sentence_transformer", "andy"] else 1536

        if self.model_type in ["openai", "openrouter"]:
            return self._create_api_embeddings(
                texts, batch_size, dimensions, show_progress_bar, **kwargs
            )
        elif self.model_type == "sentence_transformer":
            return self._create_st_embeddings(texts, batch_size, show_progress_bar, **kwargs)
        elif self.model_type == "andy":
            return self._create_andy_embeddings(texts, batch_size, show_progress_bar, **kwargs)

    def _create_api_embeddings(self, texts, batch_size, dimensions, show_progress_bar, **kwargs):
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):
            embedding_args = {"input": batch_texts, "model": self.model_name}
            if dimensions is not None and self.model_type == "openai":
                embedding_args["dimensions"] = dimensions

            embedding_args.update(kwargs)
            response = self._client.embeddings.create(**embedding_args)
            response = [data.embedding for data in response.data]
            output.extend(response)
        return self.normalize_embeddings(np.array(output))

    def _create_st_embeddings(self, texts, batch_size, show_progress_bar, **kwargs):
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):
            embeddings = self._model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, **kwargs)
            embeddings = self.normalize_embeddings(embeddings)
            output.extend(embeddings)
        return np.array(output)


    def _create_andy_embeddings(self, texts, batch_size, show_progress_bar, **kwargs):
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):
            inputs = self._tokenizer(batch_texts, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            hidden_states = outputs.last_hidden_state  # Shape: [Batch_Size, Sequence_Length, Hidden_Dimension]
            attention_mask = inputs["attention_mask"]  # Shape: [Batch_Size, Sequence_Length]

            # ==================== MASKED MEAN POOLING LOGIK ====================
            # 1. Die Attention Mask von [Batch, Seq] auf [Batch, Seq, 1] erweitern unsqueezen
            expanded_mask = attention_mask.unsqueeze(-1)

            # 2. Multipliziere die Hidden States mit der Maske.
            # Dadurch werden alle Vektoren, die zum Padding gehören, exakt auf 0 gesetzt.
            masked_hidden = hidden_states * expanded_mask

            # 3. Summiere die Vektoren über die Sequenz-Dimension (Dimension 1) auf.
            # Ergebnis-Shape: [Batch_Size, Hidden_Dimension]
            sum_embeddings = torch.sum(masked_hidden, dim=1)

            # 4. Zähle, wie viele echte (Nicht-Padding) Tokens jede Episode tatsächlich hatte.
            # keepdim=True sorgt für Shape [Batch_Size, 1] für sauberes Broadcasting.
            # clamp(min=1) schützt vor einer Division durch 0 bei komplett leeren Strings.
            token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)

            # 5. Berechne den echten Mittelwert (Summe der echten Tokens / Anzahl der echten Tokens)
            embeddings = sum_embeddings / token_counts
            # ===================================================================

            # Konvertierung zu float32 NumPy und Normalisierung für echten Cosine-Vergleich
            embeddings = embeddings.cpu().to(torch.float32).numpy()
            embeddings = self.normalize_embeddings(embeddings)

            output.extend(embeddings)
        return np.array(output)


    """
    def _create_andy_embeddings(self, texts, batch_size, show_progress_bar, **kwargs):
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):
            inputs = self._tokenizer(batch_texts, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            # Last-Token-Pooling
            last_token_indices = attention_mask.sum(dim=1) - 1
            current_batch_size = hidden_states.size(0)
            batch_indices = torch.arange(current_batch_size, device=self.device)

            embeddings = hidden_states[batch_indices, last_token_indices]
            embeddings = embeddings.cpu().to(torch.float32).numpy()
            embeddings = self.normalize_embeddings(embeddings)

            output.extend(embeddings)
        return np.array(output)"""

    def batch_generator(self, texts, batch_size, show_progress_bar=False):
        total_batches = int(np.ceil(len(texts) / batch_size))
        if show_progress_bar:
            # Sorgt dafür, dass in der Progress-Bar nur "Andy-4.2-Micro" statt des langen Pfades steht
            model_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            batch_range = tqdm(range(total_batches), desc=f"Creating {model_name} embeddings")
        else:
            batch_range = range(total_batches)

        for batch in batch_range:
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, len(texts))
            yield texts[batch_start:batch_end]

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms


"""
# Instanziierung mit dem exakten Pfad auf deinem System
embedder = EmbeddingModel(
    model_type="andy",
    model_name="/home/benito/PycharmProjects/Holist 2.0/models/Mindcraft-CE/Andy-4.2-Micro",
    device="cuda"  # Nutzt deine GPU für schnelles Last-Token-Pooling
)

# Testlauf
test_vec = embedder.create("The bot has mined one cobblestone.")
print(f"Erfolgreich geladen! Vektor-Dimension: {embedder.get_embedding_dimension()}")"""
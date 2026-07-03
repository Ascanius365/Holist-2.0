import sys

import torch
from nltk.tokenize import word_tokenize

#from consolidation.embedding import EmbeddingModel
from consolidation.Andy import EmbeddingModel
from consolidation.io import read_pickle, save_pickle
from consolidation.schema import Session

# from premem_module.src.memory import MemoryCompressor, ConversationSegmentor

# Wir müssen die Pfade hinzufügen, damit Python die PREMem-Klassen findet
sys.path.append(".")

import os
import numpy as np


def ensure_nltk_resources():
    import nltk

    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
        ("stopwords", "corpora/stopwords"),
    ]

    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


class EmbeddingDB:
    def __init__(
        self,
        dataset_name: str,
        embedding_model_name: str,
        mode: str,  # One of "session", "turn", "segment", "segment_compressed"
        model_type: str = "sentence_transformer",
        device: str = "cuda",
        base_cache_dir: str = ".cache",
        data_dir: str = "premem_module/dataset/processed",  # Default path where session pkl is stored
        batch_size: int = 128,
        show_progress_bar: bool = True,
        compress_rate: float = 0.9,
    ):

        self.dataset_name = dataset_name
        #self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        #self.embedding_model_name = "intfloat/e5-large-v2"
        self.embedding_model_name = "baai/bge-large-en-v1.5"
        self.mode = mode
        self.model_type = "openrouter"
        self.device = device
        self.base_cache_dir = base_cache_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.compressor = None
        self.compress_rate = compress_rate
        self.id2index = {}
        self.id2text = {}
        self.embeddings = {}


    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        await self.async_init()
        return self

    async def async_init(self):
        # Create embedder
        """
        self.embedder = EmbeddingModel(
            model_type=self.model_type,
            model_name=self.embedding_model_name,
            device=self.device,
        )"""

        self.embedder = EmbeddingModel(
            model_type="andy",
            model_name="/home/benito/PycharmProjects/Holist 2.0/models/Mindcraft-CE/Andy-4.2-Micro",
            device="cpu"
        )

    def retrieve(self, query_text, k=5, question_id=None, **kwargs):
        """
        Perform embedding or BM25-based search depending on the model

        Args:
            query_text (str): Text query to search for
            k (int): Number of results to return

        Returns:
            list: List of dictionaries containing search results
                 [{'id': id, 'text': text, 'score': score}, ...]
        """
        return self._retrieve_embedding(query_text, k, question_id=question_id, **kwargs)

    def _retrieve_embedding(self, query_text, k=5, question_id=None, **kwargs):
        """Embedding-based search"""
        # Convert input text to embedding
        if "stella_en_" in self.embedding_model_name:
            kwargs["prompt"] = "s2p_query"
        query_embedding = torch.tensor(
            self.embedder.create([query_text], show_progress_bar=False, **kwargs)[0]
        )

        query_embedding_normalized = query_embedding / query_embedding.norm()
        if question_id is not None:
            all_embeddings_normalized = self.embeddings[question_id] / self.embeddings[
                question_id
            ].norm(dim=1, keepdim=True)
        else:
            all_embeddings_normalized = self.embeddings / self.embeddings.norm(
                dim=1, keepdim=True
            )

        scores = all_embeddings_normalized @ query_embedding_normalized.T
        top_indices = torch.topk(scores, min(k, len(scores))).indices

        # Compile results
        results = []
        if question_id is not None:
            index_to_id = {idx: id_ for id_, idx in self.id2index[question_id].items()}
        else:
            index_to_id = {idx: id_ for id_, idx in self.id2index.items()}

        for idx in top_indices:
            idx = idx.item()
            result_id = index_to_id[idx]
            if question_id is not None:
                result_text = self.id2text[question_id][result_id]
            else:
                result_text = self.id2text[result_id]

            result_dict = {"id": result_id, **result_text, "score": float(scores[idx])}

            results.append(result_dict)

        return results
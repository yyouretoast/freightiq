import os
import time
import logging
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

logger = logging.getLogger(__name__)

# Singletons and Thread Locks
_EMBED_MODEL = None
_CROSS_ENCODER = None

_embed_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _embed_lock:
            if _EMBED_MODEL is None:
                logger.info(f"Loading SentenceTransformer model: {config.EMBEDDING_MODEL_NAME}")
                _EMBED_MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL

def get_cross_encoder():
    """
    Cached singleton for the cross-encoder reranker.
    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        with _cross_encoder_lock:
            if _CROSS_ENCODER is None:
                model_name = getattr(config, "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                try:
                    logger.info(f"Loading CrossEncoder model: {model_name}")
                    _CROSS_ENCODER = CrossEncoder(model_name)
                except Exception as e:
                    logger.error(f"Failed to load CrossEncoder ({e}). Will fall back to cosine similarity.")
                    _CROSS_ENCODER = None
    return _CROSS_ENCODER

def rerank_documents(query, documents, metadatas=None, top_k=5, doc_embeddings=None, query_embedding=None, force_cosine=False):
    """
    Reranks candidate documents using Cross-Encoder attention scoring.
    Falls back to dense embedding cosine similarity if force_cosine is True or if CrossEncoder fails.
    """
    if not documents:
        return []

    scores = None
    if not force_cosine:
        cross_encoder = get_cross_encoder()
        if cross_encoder is not None:
            try:
                pairs = [[query, doc] for doc in documents]
                scores = cross_encoder.predict(pairs)
                scores = np.array(scores)
                logger.info("Reranked candidate documents utilizing pre-trained Cross-Encoder.")
            except Exception as e:
                logger.error(f"CrossEncoder prediction failed ({e}), falling back to cosine similarity.")
                scores = None

    if scores is None:
        # Fallback to cosine similarity with sentence-transformer embeddings
        embed_model = get_embed_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if query_embedding is None:
            query_vector = embed_model.encode(query, convert_to_numpy=True)
        else:
            query_vector = np.array(query_embedding)

        if doc_embeddings is None:
            doc_vectors = embed_model.encode(documents, convert_to_numpy=True)
        else:
            doc_vectors = np.array(doc_embeddings)

        query_tensor = torch.tensor(query_vector, dtype=torch.float32, device=device).unsqueeze(0)
        doc_tensors = torch.tensor(doc_vectors, dtype=torch.float32, device=device)
        query_tensors = query_tensor.expand(len(documents), -1)

        with torch.no_grad():
            scores = F.cosine_similarity(query_tensors, doc_tensors, dim=-1).cpu().numpy()
        logger.info("Reranked candidate documents utilizing dense cosine similarity fallback.")

    ranked_indices = np.argsort(scores)[::-1]
    logger.debug(f"Reranked {len(documents)} docs, top score: {scores[ranked_indices[0]]:.4f}")

    return [
        {
            "document": documents[idx],
            "score": float(scores[idx]),
            "metadata": metadatas[idx] if metadatas else {}
        }
        for idx in ranked_indices[:top_k]
    ]

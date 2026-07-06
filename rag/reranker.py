import os
import logging
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)

# Singletons and Thread Locks
_EMBED_MODEL = None
_RERANKER_MODEL = None
_RERANKER_LOADED_TIME = 0.0

_embed_lock = threading.Lock()
_reranker_lock = threading.Lock()

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _embed_lock:
            if _EMBED_MODEL is None:
                logger.info(f"Loading SentenceTransformer model: {config.EMBEDDING_MODEL_NAME}")
                _EMBED_MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL

def get_reranker_model(device):
    global _RERANKER_MODEL, _RERANKER_LOADED_TIME
    weights_path = os.path.join(config.BASE_DIR, "rag", "data", "reranker_weights.pt")
    if not os.path.exists(weights_path):
        return None
        
    with _reranker_lock:
        mtime = os.path.getmtime(weights_path)
        if _RERANKER_MODEL is None or mtime > _RERANKER_LOADED_TIME:
            logger.info("Initializing/Reloading CarrierReRanker weights from disk...")
            if _RERANKER_MODEL is None:
                _RERANKER_MODEL = CarrierReRanker(
                    embedding_dim=config.EMBEDDING_DIM, 
                    hidden_dim=config.RERANKER_HIDDEN_DIM
                ).to(device)
            _RERANKER_MODEL.load_state_dict(torch.load(weights_path, map_location=device))
            _RERANKER_MODEL.eval()
            _RERANKER_LOADED_TIME = mtime
    return _RERANKER_MODEL

class CarrierReRanker(nn.Module):
    """
    2-layer MLP reranker architecture. Intended for future supervised training
    on broker-carrier match logs using BCE loss. Currently, scoring is performed
    via cosine similarity (see rerank_documents) pending training data collection.
    """
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, query_emb, doc_emb):
        x = torch.cat((query_emb, doc_emb), dim=-1)
        return self.mlp(x)

def rerank_documents(query, documents, metadatas=None, top_k=5, doc_embeddings=None, query_embedding=None, force_cosine=False):
    if not documents:
        return []

    embed_model = get_embed_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dynamic Reranker Execution:
    # Attempt to load the fine-tuned MLP weights if they exist on disk.
    # If the weights file is absent, score candidate documents using cosine similarity.

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

    model = None if force_cosine else get_reranker_model(device)
    if model is not None:
        try:
            with torch.no_grad():
                scores_tensor = model(query_tensors, doc_tensors).squeeze(-1)
                scores = scores_tensor.cpu().numpy()
            logger.info("Reranked candidate documents utilizing fine-tuned PyTorch MLP reranker.")
        except Exception as e:
            logger.error(f"Failed to run custom PyTorch reranker, falling back to cosine similarity: {e}")
            with torch.no_grad():
                scores = F.cosine_similarity(query_tensors, doc_tensors, dim=-1).cpu().numpy()
    else:
        # Score via cosine similarity — deterministic and semantically correct.
        with torch.no_grad():
            scores = F.cosine_similarity(query_tensors, doc_tensors, dim=-1).cpu().numpy()

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

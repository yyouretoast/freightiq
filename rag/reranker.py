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
    """
    Placeholder initialization hook for the neural network reranker.
    Preserved for future active-learning training integration; currently unused in active
    inference to optimize memory footprint in container builds.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        with _reranker_lock:
            if _RERANKER_MODEL is None:
                logger.info(f"Initializing CarrierReRanker on device: {device}")
                _RERANKER_MODEL = CarrierReRanker(
                    embedding_dim=config.EMBEDDING_DIM, 
                    hidden_dim=config.RERANKER_HIDDEN_DIM
                ).to(device)
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

def rerank_documents(query, documents, metadatas=None, top_k=5, doc_embeddings=None, query_embedding=None):
    if not documents:
        return []

    embed_model = get_embed_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reranker model instantiation removed to optimize CPU/GPU memory footprint.
    # The CarrierReRanker class is preserved above for architectural completeness and future training.

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

    # Check if custom fine-tuned weights exist to enable active learning inference path
    weights_path = os.path.join(config.BASE_DIR, "rag", "data", "reranker_weights.pt")
    if os.path.exists(weights_path):
        try:
            model = get_reranker_model(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            with torch.no_grad():
                scores_tensor = model(query_tensors, doc_tensors).squeeze(-1)
                scores = scores_tensor.cpu().numpy()
            logger.info("Reranked candidate documents utilizing fine-tuned PyTorch MLP reranker.")
        except Exception as e:
            logger.error(f"Failed to load/run custom PyTorch reranker, falling back to cosine similarity: {e}")
            with torch.no_grad():
                scores = F.cosine_similarity(query_tensors, doc_tensors, dim=-1).cpu().numpy()
    else:
        # Score via cosine similarity — deterministic and semantically correct.
        with torch.no_grad():
            scores = F.cosine_similarity(query_tensors, doc_tensors, dim=-1)
            scores = scores.cpu().numpy()

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

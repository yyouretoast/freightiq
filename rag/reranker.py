import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = None
_RERANKER_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL

def get_reranker_model(device):
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        _RERANKER_MODEL = CarrierReRanker(embedding_dim=384, hidden_dim=128).to(device)
    return _RERANKER_MODEL

class CarrierReRanker(nn.Module):
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
    
    reranker = get_reranker_model(device)
    reranker.eval()
    
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
    query_tensors = query_tensor.repeat(len(documents), 1)
    
    with torch.no_grad():
        scores = reranker(query_tensors, doc_tensors)
        scores = scores.squeeze(-1).cpu().numpy()
        
    ranked_indices = np.argsort(scores)[::-1]
    
    return [
        {
            "document": documents[idx],
            "score": float(scores[idx]),
            "metadata": metadatas[idx] if metadatas else {}
        }
        for idx in ranked_indices[:top_k]
    ]

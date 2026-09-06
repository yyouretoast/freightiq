import sys
import os

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import config
from rag.reranker import get_embed_model
from rag.utils import format_carrier_document, load_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class CarrierReRanker(nn.Module):
    """
    2-layer MLP reranker architecture.
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

def generate_bootstrap_data(carriers):
    """
    Generate synthetic query-document pairs directly from carrier attributes
    to bootstrap model training in the absence of sufficient user feedback.
    """
    logger.info("Generating bootstrap query-document pairs...")
    bootstrap_pairs = []
    
    # Define query templates
    for c in carriers:
        doc = format_carrier_document(c)
        
        # 1. State queries
        q_state = f"Find a carrier headquartered in {c['hq_state']}"
        bootstrap_pairs.append((q_state, doc, 1.0))
        # Negative state match
        other_states = [other for other in carriers if other['hq_state'] != c['hq_state']]
        if other_states:
            neg_c = random.choice(other_states)
            bootstrap_pairs.append((q_state, format_carrier_document(neg_c), 0.0))
            
        # 2. Equipment queries
        for eq in c['equipment_types']:
            q_eq = f"We need a carrier with {eq} equipment"
            bootstrap_pairs.append((q_eq, doc, 1.0))
            # Negative equipment match
            other_eqs = [other for other in carriers if eq not in other['equipment_types']]
            if other_eqs:
                neg_c = random.choice(other_eqs)
                bootstrap_pairs.append((q_eq, format_carrier_document(neg_c), 0.0))
                
        # 3. Specialization queries
        for spec in c['cargo_specializations']:
            q_spec = f"Show me carriers that specialize in {spec}"
            bootstrap_pairs.append((q_spec, doc, 1.0))
            # Negative spec match
            other_specs = [other for other in carriers if spec not in other['cargo_specializations']]
            if other_specs:
                neg_c = random.choice(other_specs)
                bootstrap_pairs.append((q_spec, format_carrier_document(neg_c), 0.0))

        # 4. Safety rating queries
        q_safety = f"Find a carrier with a {c['safety_rating']} safety rating"
        bootstrap_pairs.append((q_safety, doc, 1.0))
        # Negative safety match
        other_safeties = [other for other in carriers if other['safety_rating'] != c['safety_rating']]
        if other_safeties:
            neg_c = random.choice(other_safeties)
            bootstrap_pairs.append((q_safety, format_carrier_document(neg_c), 0.0))

    return bootstrap_pairs

def load_feedback_data(carriers):
    """
    Load user feedback from feedback.json/feedback.jsonl and parse it into query-document pairs.
    """
    data = load_feedback()
    if not data:
        logger.info("No user feedback records found.")
        return []
        
    feedback_pairs = []
    logger.info(f"Parsing {len(data)} feedback records...")
    
    for record in data:
        query = record.get("query", "")
        response = record.get("response", "")
        feedback = record.get("feedback", "")
        
        # Exclude mock stress-test records
        if "Test Query from worker" in query:
            continue
        
        if not query or not response or not feedback:
            continue
            
        # Determine label based on thumbs up/down
        label = 1.0 if feedback == "up" else 0.0
        
        # Identify which carriers were mentioned in the response
        matched_carriers = []
        for c in carriers:
            if c["carrier_name"] in response or c["dot_number"] in response:
                matched_carriers.append(c)
                
        for mc in matched_carriers:
            doc = format_carrier_document(mc)
            feedback_pairs.append((query, doc, label))
            
            if label == 1.0:
                unmatched = [other for other in carriers if other not in matched_carriers]
                if unmatched:
                    neg_c = random.choice(unmatched)
                    feedback_pairs.append((query, format_carrier_document(neg_c), 0.0))
                    
    logger.info(f"Extracted {len(feedback_pairs)} query-document pairs from feedback logs.")
    return feedback_pairs

def main():
    logger.info("=== Starting PyTorch CarrierReRanker Training Pipeline ===")
    
    # Load carrier database
    json_path = config.CARRIERS_JSON_PATH
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Source carriers.json not found at {json_path}. Run setup.py first.")
        
    with open(json_path, "r", encoding="utf-8") as f:
        carriers = json.load(f)
        
    # Split carriers into disjoint train (80%) and validation (20%) sets to prevent data leakage
    random.seed(42)
    shuffled_carriers = list(carriers)
    random.shuffle(shuffled_carriers)
    split_idx = int(0.8 * len(shuffled_carriers))
    train_carriers = shuffled_carriers[:split_idx]
    val_carriers = shuffled_carriers[split_idx:]
    
    train_bootstrap = generate_bootstrap_data(train_carriers)
    val_bootstrap = generate_bootstrap_data(val_carriers)
    
    # Load feedback data
    real_pairs = load_feedback_data(carriers)
    random.shuffle(real_pairs)
    real_split = int(0.8 * len(real_pairs))
    train_pairs = real_pairs[:real_split] + train_bootstrap
    val_pairs = real_pairs[real_split:] + val_bootstrap
    
    logger.info(f"Total dataset: {len(train_pairs)} train pairs | {len(val_pairs)} val pairs.")
    
    embed_model = get_embed_model()
    
    logger.info("Generating training embeddings...")
    X_query_train = torch.tensor(embed_model.encode([p[0] for p in train_pairs], show_progress_bar=True, convert_to_numpy=True), dtype=torch.float32)
    X_doc_train = torch.tensor(embed_model.encode([p[1] for p in train_pairs], show_progress_bar=True, convert_to_numpy=True), dtype=torch.float32)
    y_train = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32).unsqueeze(1)
    
    logger.info("Generating validation embeddings...")
    X_query_val = torch.tensor(embed_model.encode([p[0] for p in val_pairs], show_progress_bar=True, convert_to_numpy=True), dtype=torch.float32)
    X_doc_val = torch.tensor(embed_model.encode([p[1] for p in val_pairs], show_progress_bar=True, convert_to_numpy=True), dtype=torch.float32)
    y_val = torch.tensor([p[2] for p in val_pairs], dtype=torch.float32).unsqueeze(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device} | Train Size: {len(train_pairs)} | Val Size: {len(val_pairs)}")
    
    train_dataset = TensorDataset(X_query_train, X_doc_train, y_train)
    val_dataset = TensorDataset(X_query_val, X_doc_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize ReRanker model
    model = CarrierReRanker(embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.RERANKER_HIDDEN_DIM).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model with early stopping
    epochs = 15
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    logger.info(f"Training MLP model for up to {epochs} epochs (early stopping patience={patience})...")
    
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        epoch_train_loss = 0.0
        for batch_query, batch_doc, batch_label in train_loader:
            batch_query = batch_query.to(device)
            batch_doc = batch_doc.to(device)
            batch_label = batch_label.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_query, batch_doc)
            loss = criterion(predictions, batch_label)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_query.size(0)
            
        avg_train_loss = epoch_train_loss / len(train_dataset)
        
        # 2. Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_query, batch_doc, batch_label in val_loader:
                batch_query = batch_query.to(device)
                batch_doc = batch_doc.to(device)
                batch_label = batch_label.to(device)
                
                predictions = model(batch_query, batch_doc)
                loss = criterion(predictions, batch_label)
                epoch_val_loss += loss.item() * batch_query.size(0)
                
        avg_val_loss = epoch_val_loss / len(val_dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # Early stopping: save best model and track improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            logger.info(f"  ↳ New best val loss: {best_val_loss:.5f} — checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")
                break
            
    # Save the best model weights
    if best_model_state is None:
        best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    weights_path = config.WEIGHTS_PATH
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    temp_weights_path = f"{weights_path}.tmp"
    torch.save(best_model_state, temp_weights_path)
    os.replace(temp_weights_path, weights_path)
    logger.info(f"Best model weights (val loss={best_val_loss:.5f}) atomically saved to {weights_path}")
    logger.info("=== Reranker Training Pipeline Complete ===")

if __name__ == "__main__":
    main()

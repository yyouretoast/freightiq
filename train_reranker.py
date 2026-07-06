import os
import json
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import config
from rag.reranker import CarrierReRanker, get_embed_model
from rag.utils import format_carrier_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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
    Load user feedback from feedback.json and parse it into query-document pairs.
    """
    feedback_path = os.path.join(config.BASE_DIR, "rag", "data", "feedback.json")
    if not os.path.exists(feedback_path):
        logger.info("No feedback.json file found.")
        return []
        
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read feedback.json: {e}")
        return []
        
    feedback_pairs = []
    logger.info(f"Parsing {len(data)} feedback records...")
    
    for record in data:
        query = record.get("query", "")
        response = record.get("response", "")
        feedback = record.get("feedback", "")
        
        if not query or not response or not feedback:
            continue
            
        # Determine label based on thumbs up/down
        label = 1.0 if feedback == "up" else 0.0
        
        # Identify which carriers were mentioned in the response
        matched_carriers = []
        for c in carriers:
            # Check if name or DOT number is mentioned in response
            if c["carrier_name"] in response or c["dot_number"] in response:
                matched_carriers.append(c)
                
        for mc in matched_carriers:
            doc = format_carrier_document(mc)
            feedback_pairs.append((query, doc, label))
            
            # Negative sampling: if positive feedback, add a random negative sample
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
        
    # Load feedback and bootstrap dataset
    real_pairs = load_feedback_data(carriers)
    bootstrap_pairs = generate_bootstrap_data(carriers)
    
    # Combine datasets (weighing real feedback higher if present)
    all_pairs = real_pairs + bootstrap_pairs
    logger.info(f"Total dataset size: {len(all_pairs)} query-document pairs.")
    
    # Deduplicate and split queries / docs
    queries = [p[0] for p in all_pairs]
    documents = [p[1] for p in all_pairs]
    labels = [p[2] for p in all_pairs]
    
    # Load SentenceTransformer model to generate embeddings
    embed_model = get_embed_model()
    
    logger.info("Generating query embeddings...")
    query_embs = embed_model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    
    logger.info("Generating document embeddings...")
    doc_embs = embed_model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    # Convert to PyTorch tensors
    X_query = torch.tensor(query_embs, dtype=torch.float32)
    X_doc = torch.tensor(doc_embs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    # Split into train/validation datasets (80% train, 20% validation)
    dataset_size = len(all_pairs)
    indices = list(range(dataset_size))
    random.seed(42)  # For reproducible splits
    random.shuffle(indices)
    
    split_idx = int(0.8 * dataset_size)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_query_train, X_query_val = X_query[train_indices], X_query[val_indices]
    X_doc_train, X_doc_val = X_doc[train_indices], X_doc[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device} | Train Size: {len(train_indices)} | Val Size: {len(val_indices)}")
    
    train_dataset = TensorDataset(X_query_train, X_doc_train, y_train)
    val_dataset = TensorDataset(X_query_val, X_doc_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize ReRanker model
    model = CarrierReRanker(embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.RERANKER_HIDDEN_DIM).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    epochs = 15
    logger.info(f"Training MLP model for {epochs} epochs...")
    
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
            
    # Save trained weights
    out_dir = os.path.join(config.BASE_DIR, "rag", "data")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "reranker_weights.pt")
    
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Model weights saved successfully to {weights_path}")
    logger.info("=== Reranker Training Pipeline Complete ===")

if __name__ == "__main__":
    main()

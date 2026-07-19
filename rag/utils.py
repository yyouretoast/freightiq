import os
import json
import logging
from datetime import datetime, timezone
import config
from agent.locks import feedback_lock

logger = logging.getLogger(__name__)

def format_carrier_document(c):
    return (
        f"Carrier Name: {c['carrier_name']}\n"
        f"DOT Number: {c['dot_number']}\n"
        f"MC Number: {c['mc_number']}\n"
        f"HQ State: {c['hq_state']}\n"
        f"Service Regions: {', '.join(c['service_regions'])}\n"
        f"Equipment: {', '.join(c['equipment_types'])}\n"
        f"Specializations: {', '.join(c['cargo_specializations'])}\n"
        f"Safety Rating: {c['safety_rating']}\n"
        f"Years Operating: {c['years_operating']} years\n"
        f"Contact: {c['contact_email']}\n"
        f"Notes: {c['notes']}"
    )

def format_message_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)

def save_feedback(query, response, feedback_type):
    feedback_file = os.path.join(config.BASE_DIR, "rag", "data", "feedback.json")
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": format_message_content(query),
        "response": format_message_content(response),
        "feedback": feedback_type
    }
    
    with feedback_lock:
        data = []
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing feedback.json ({e}). Re-initializing feedback file.")
                
        data.append(record)
        try:
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write feedback record: {e}")

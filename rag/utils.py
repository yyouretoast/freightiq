import os
import json
import logging
from datetime import datetime, timezone
import config
from utils.locks import feedback_lock

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
def load_feedback(filepath=None):
    feedback_file = filepath or config.FEEDBACK_PATH
    if not os.path.exists(feedback_file):
        return []
    
    with feedback_lock:
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return []
            if content.startswith("["):
                try:
                    return json.loads(content)
                except Exception:
                    pass
            records = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("[") and not line.startswith("]"):
                    try:
                        records.append(json.loads(line.rstrip(",")))
                    except Exception:
                        continue
            return records
        except Exception as e:
            logger.error(f"Failed to read feedback from {feedback_file}: {e}")
            return []

def save_feedback(query, response, feedback_type, filepath=None):
    feedback_file = filepath or config.FEEDBACK_PATH
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": format_message_content(query),
        "response": format_message_content(response),
        "feedback": feedback_type
    }
    
    record_line = json.dumps(record, ensure_ascii=False)
    
    with feedback_lock:
        try:
            # If existing file is a JSON array, migrate to JSONL format once
            if os.path.exists(feedback_file):
                with open(feedback_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content.startswith("["):
                    try:
                        existing = json.loads(content)
                        if isinstance(existing, list):
                            with open(feedback_file, "w", encoding="utf-8") as f:
                                for item in existing:
                                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
            
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(record_line + "\n")
        except Exception as e:
            logger.error(f"Failed to write feedback record: {e}")

import sqlite3
import json
import os

def setup_sqlite():
    json_path = os.path.join("rag", "data", "carriers.json")
    db_path = os.path.join("rag", "data", "carriers.db")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Source carriers.json not found at {json_path}. Run generate_carriers.py first.")
        
    print("Setting up SQLite database...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS carriers")
        cursor.execute("""
        CREATE TABLE carriers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            carrier_name TEXT NOT NULL,
            dot_number TEXT UNIQUE NOT NULL,
            mc_number TEXT UNIQUE NOT NULL,
            hq_state TEXT NOT NULL,
            service_regions TEXT NOT NULL,
            equipment_types TEXT NOT NULL,
            cargo_specializations TEXT NOT NULL,
            safety_rating TEXT NOT NULL,
            years_operating INTEGER NOT NULL,
            contact_email TEXT NOT NULL,
            notes TEXT NOT NULL
        )
        """)
        
        with open(json_path, "r") as f:
            carriers = json.load(f)
            
        batch_data = [
            (
                c["carrier_name"],
                c["dot_number"],
                c["mc_number"],
                c["hq_state"],
                ", ".join(c["service_regions"]),
                ", ".join(c["equipment_types"]),
                ", ".join(c["cargo_specializations"]),
                c["safety_rating"],
                c["years_operating"],
                c["contact_email"],
                c["notes"]
            )
            for c in carriers
        ]
        
        cursor.executemany("""
        INSERT INTO carriers (
            carrier_name, dot_number, mc_number, hq_state, 
            service_regions, equipment_types, cargo_specializations, 
            safety_rating, years_operating, contact_email, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_data)
        
    print(f"Successfully populated SQLite database with {len(carriers)} carrier profiles at {db_path}.")

if __name__ == "__main__":
    setup_sqlite()

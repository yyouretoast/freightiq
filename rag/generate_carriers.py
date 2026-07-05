import json
import random
import os
import config

STATES = ["TX", "CA", "IL", "OH", "PA", "GA", "FL", "NC", "MI", "NY", "TN", "IN", "KY", "AL", "MO", "AZ", "WI", "CO", "SC", "OR"]
REGIONS = ["Midwest", "Northeast", "Southeast", "Southwest", "West Coast", "Pacific Northwest", "Mountain"]
EQUIPMENT = ["dry van", "flatbed", "reefer", "tanker", "LTL", "intermodal"]
SPECIALIZATIONS = ["hazardous materials", "oversized loads", "fresh produce", "automotive parts", "pharmaceuticals", "retail goods", "building materials", "electronics", "machinery", "general freight"]
SAFETY_RATINGS = ["satisfactory", "satisfactory", "satisfactory", "conditional", "unsatisfactory"]
COMPANY_PREFIXES = ["Swift", "Apex", "Blue Ribbon", "Crossroads", "Falcon", "Ironclad", "Golden Gate", "Red Line", "Titan", "Pioneer", "Liberty", "Voyager", "Interstate", "NextGen", "Summit", "FreightRunner", "Atlas", "Canyon", "Echo", "Express"]
COMPANY_SUFFIXES = ["Logistics", "Transport", "Freight", "Trucking", "Carriers", "Lines", "Express", "Systems", "Haulers", "Solutions"]

def generate_random_carrier(i):
    prefix = random.choice(COMPANY_PREFIXES)
    suffix = random.choice(COMPANY_SUFFIXES)
    name = f"{prefix} {suffix} {random.randint(10, 99)}" if random.random() > 0.5 else f"{prefix} {suffix}"
    name = f"{name} #{i+100}"
    
    dot = f"{random.randint(1000000, 9999999)}"
    mc = f"{random.randint(100000, 999999)}"
    hq = random.choice(STATES)
    
    num_regions = random.randint(1, 3)
    serv_regions = list(set(random.choice(REGIONS) for _ in range(num_regions)))
    
    num_equip = random.randint(1, 3)
    equip = list(set(random.choice(EQUIPMENT) for _ in range(num_equip)))
    
    num_specs = random.randint(1, 2)
    specs = list(set(random.choice(SPECIALIZATIONS) for _ in range(num_specs)))
    
    safety = random.choice(SAFETY_RATINGS)
    years = random.randint(2, 25)
    
    email_name = name.lower().replace(" ", "").replace("#", "")
    email = f"info@{email_name}.com"
    
    equip_str = ", ".join(equip)
    spec_str = ", ".join(specs)
    notes = f"Specializes in {spec_str} hauling with primary equipment using {equip_str}. Known for reliable shipping routes across {', '.join(serv_regions)}."
    
    return {
        "carrier_name": name,
        "dot_number": dot,
        "mc_number": mc,
        "hq_state": hq,
        "service_regions": serv_regions,
        "equipment_types": equip,
        "cargo_specializations": specs,
        "safety_rating": safety,
        "years_operating": years,
        "contact_email": email,
        "notes": notes
    }

def main():
    # Seed random generator for reproducibility
    random.seed(42)
    carriers = [generate_random_carrier(i) for i in range(200)]
    out_path = config.CARRIERS_JSON_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(carriers, f, indent=4)
    print(f"Generated 200 synthetic carrier profiles at {out_path}")

if __name__ == "__main__":
    main()

import json
import random
import os
import config

STATES = ["TX", "CA", "IL", "OH", "PA", "GA", "FL", "NC", "MI", "NY", "TN", "IN", "KY", "AL", "MO", "AZ", "WI", "CO", "SC", "OR"]
REGIONS = ["Midwest", "Northeast", "Southeast", "Southwest", "West Coast", "Pacific Northwest", "Mountain"]
EQUIPMENT = ["dry van", "flatbed", "reefer", "tanker", "LTL", "intermodal"]
SPECIALIZATIONS = ["hazardous materials", "oversized loads", "fresh produce", "automotive parts", "pharmaceuticals", "retail goods", "building materials", "electronics", "machinery", "general freight"]
SAFETY_RATINGS = ["satisfactory", "satisfactory", "satisfactory", "conditional", "unsatisfactory"]

COMPANY_NAMES = [
    ("Swift Transportation Services", "DBA Swift Intermodal"),
    ("Knight Carrier Systems", "DBA Knight Cold Chain"),
    ("Schneider National Dedicated", "DBA Schneider Bulk"),
    ("J.B. Hunt Transport Solutions", "DBA J.B. Hunt Final Mile"),
    ("Werner Logistics & Transport", "DBA Werner Enterprises"),
    ("Old Dominion Freight Line", None),
    ("Estes Express Lines Regional", None),
    ("R+L Carriers Dedicated", None),
    ("Saia Motor Freight Line", None),
    ("Landstar Ranger Fleet", "DBA Landstar Inway"),
    ("Marten Transport Cold Chain", None),
    ("Stevens Transport Reefer Fleet", None),
    ("Prime Inc. Refrigerated & Flatbed", None),
    ("Covenant Logistics Group", None),
    ("Heartland Express Dedicated", None),
    ("A. Duie Pyle LTL & Custom", None),
    ("Southeastern Freight Lines", None),
    ("Hub Group Dedicated Drayage", None),
    ("Forward Air Truckload", None),
    ("KLLM Transport Services", None),
]

NAME_PREFIXES = [
    "Apex", "Blue Ribbon", "Crossroads", "Falcon", "Ironclad", "Golden Gate",
    "Red Line", "Titan", "Pioneer", "Liberty", "Voyager", "Interstate",
    "NextGen", "Summit", "FreightRunner", "Atlas", "Canyon", "Echo", "Express",
    "Allegheny", "Sunbelt", "Buckeye", "Keystone", "Lone Star", "Palmetto",
    "Great Lakes", "Cascade", "Piedmont", "Shenandoah", "Ozark", "Prairie",
    "Empire", "Bay State", "Frontier", "Badger", "Wolverine", "Hoosier"
]

NAME_SUFFIXES = [
    "Logistics LLC", "Transport Inc.", "Freight Corp.", "Trucking Co.",
    "Carriers LLC", "Lines Inc.", "Express Logistics LLC", "Systems Inc.",
    "Haulers LLC", "Solutions Corp.", "Intermodal Services LLC",
    "Supply Chain Fleet LLC", "Dedicated Freight Lines Inc."
]

SPEC_DETAILS = {
    "fresh produce": [
        "continuous pulp temperature monitoring and pre-cooled trailers for berries and leafy greens",
        "temperature-controlled produce hauling with daily automated pulp temp logs",
        "rapid farm-to-cooler transport specializing in seasonal citrus and fresh vegetables"
    ],
    "pharmaceuticals": [
        "strict temperature-controlled cold chain (-20°C to 25°C) with dual reefer units and tamper-evident seals",
        "GDP-compliant pharmaceutical freight with real-time continuous telematics and remote temperature alarms",
        "validated cold chain transport for life sciences, vaccines, and clinical trial supplies"
    ],
    "hazardous materials": [
        "Hazmat Class 3 (Flammable Liquids), Class 8 (Corrosives), and Class 9 materials with spill kits on board",
        "certified hazmat transportation with placards, certified chemical handlers, and 24/7 Chemtrec monitoring",
        "licensed hazardous waste and flammable chemical transport with emergency response protocols"
    ],
    "electronics": [
        "high-value electronics freight with dual-driver teams, high-security lockboxes, and covert GPS geofencing",
        "secure sealed transport for server racks, semiconductors, and consumer electronics",
        "bonded carrier equipped for high-theft risk consumer technology with satellite tracking"
    ],
    "building materials": [
        "heavy construction lumber, roofing shingles, and drywall transport with heavy-duty tarps and straps",
        "jobsite flatbed delivery of structural steel beams, rebar, and precast concrete components",
        "palletized masonry, stone, and architectural materials with Moffett forklift offloading"
    ],
    "machinery": [
        "oversized industrial machinery and agricultural equipment with wide-load permits and escort coordination",
        "heavy plant machinery transport using specialized drop-deck and RGN lowboy trailers with 4-point chains",
        "precision CNC equipment and manufacturing tooling with air-ride suspension and weather protection"
    ],
    "automotive parts": [
        "tier-1 automotive supplier parts with strict Just-In-Time (JIT) delivery windows to assembly plants",
        "engine blocks, chassis stampings, and transmission freight operating on round-the-clock milk-runs",
        "sequenced automotive assembly line parts distribution across the Midwest manufacturing corridor"
    ],
    "retail goods": [
        "high-cube retail freight for national distribution center networks with drop-and-hook capabilities",
        "consumer packaged goods (CPG) store delivery with liftgate offload and pallet exchange",
        "seasonal e-commerce overflow and peak retail stock replenishment"
    ],
    "oversized loads": [
        "permitted heavy-haul operations up to 120,000 lbs gross vehicle weight with multi-state routing permits",
        "superload cargo including wind turbine blades and bridge girders with certified steerable trailers",
        "dimensional oversized freight requiring specialized route surveys and police escorts"
    ],
    "general freight": [
        "general commercial freight across regional distribution lanes with 99.2% on-time delivery record",
        "palletized dry freight with clean, food-grade 53' air-ride van trailers",
        "multi-stop TL and LTL freight consolidation with live GPS milestone tracking"
    ]
}

EQUIP_DETAILS = {
    "dry van": "53ft air-ride dry vans with roll doors, wood floors, and internal E-track securement",
    "flatbed": "48ft and 53ft aluminum spread-axle flatbeds with 4-inch heavy-duty straps, chains, and 8ft drop tarps",
    "reefer": "53ft multi-temp refrigerated units with Carrier Vector/Thermo King chillers capable of -20°F to 70°F",
    "tanker": "DOT 407/412 insulated stainless steel sanitary and chemical liquid bulk tankers with rear pump-off",
    "LTL": "26ft straight trucks with 3,500 lb hydraulic liftgates and pallet jacks for regional cross-dock LTL",
    "intermodal": "intermodal chassis fleet equipped for 20ft, 40ft, and 53ft ISO container drayage with port TWIC badges"
}

def generate_random_carrier(i):
    if i < len(COMPANY_NAMES):
        base_name, dba = COMPANY_NAMES[i]
        name = f"{base_name} ({dba})" if dba else base_name
    else:
        prefix = random.choice(NAME_PREFIXES)
        suffix = random.choice(NAME_SUFFIXES)
        dba_opt = f" (DBA {prefix} Freight Solutions)" if random.random() < 0.25 else ""
        name = f"{prefix} {suffix}{dba_opt}"

    dot = str(random.randint(1000000, 3999999))
    mc = str(random.randint(100000, 999999))
    hq = random.choice(STATES)

    num_regions = random.randint(1, 3)
    serv_regions = list(set(random.choice(REGIONS) for _ in range(num_regions)))

    num_equip = random.randint(1, 3)
    equip = list(set(random.choice(EQUIPMENT) for _ in range(num_equip)))

    num_specs = random.randint(1, 2)
    specs = list(set(random.choice(SPECIALIZATIONS) for _ in range(num_specs)))

    safety = random.choice(SAFETY_RATINGS)
    years = random.randint(2, 35)

    clean_slug = "".join(c for c in name.lower() if c.isalnum())[:20]
    email = f"dispatch@{clean_slug}.com"

    # Build authentic, informative notes
    spec_notes = []
    for s in specs:
        if s in SPEC_DETAILS:
            spec_notes.append(random.choice(SPEC_DETAILS[s]))
        else:
            spec_notes.append(f"specialized {s} transport")

    equip_notes = [EQUIP_DETAILS.get(e, f"{e} equipment") for e in equip]

    notes = (
        f"Fleet operations: {'; '.join(equip_notes)}. "
        f"Cargo handling: {'; '.join(spec_notes)}. "
        f"Operating network spans {', '.join(serv_regions)} with headquarters in {hq}. "
        f"FMCSA compliance rating: {safety} with {years} years of active interstate motor carrier operations."
    )

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
    random.seed(42)
    carriers = [generate_random_carrier(i) for i in range(500)]
    out_path = config.CARRIERS_JSON_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(carriers, f, indent=4)
    print(f"Generated 500 carrier profiles at {out_path}")

if __name__ == "__main__":
    main()

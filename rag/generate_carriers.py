import json
import random
import os
import config

STATES = ["TX", "CA", "IL", "OH", "PA", "GA", "FL", "NC", "MI", "NY", "TN", "IN", "KY", "AL", "MO", "AZ", "WI", "CO", "SC", "OR"]
REGIONS = ["Midwest", "Northeast", "Southeast", "Southwest", "West Coast", "Pacific Northwest", "Mountain"]
EQUIPMENT = ["dry van", "flatbed", "reefer", "tanker", "LTL", "intermodal"]
SPECIALIZATIONS = ["hazardous materials", "oversized loads", "fresh produce", "automotive parts", "pharmaceuticals", "retail goods", "building materials", "electronics", "machinery", "general freight"]
SAFETY_RATINGS = ["satisfactory", "satisfactory", "satisfactory", "conditional", "unsatisfactory"]

NAME_PREFIXES = [
    "Apex", "Blue Ribbon", "Crossroads", "Falcon", "Ironclad", "Golden Gate",
    "Red Line", "Titan", "Pioneer", "Liberty", "Voyager", "Interstate",
    "NextGen", "Summit", "FreightRunner", "Atlas", "Canyon", "Echo", "Express",
    "Allegheny", "Sunbelt", "Buckeye", "Keystone", "Lone Star", "Palmetto",
    "Great Lakes", "Cascade", "Piedmont", "Shenandoah", "Ozark", "Prairie",
    "Empire", "Bay State", "Frontier", "Badger", "Wolverine", "Hoosier",
    "Bluegrass", "Granite", "Magnolia", "North Star", "Silver State", "Evergreen",
    "Apache", "Sequoia", "Highland", "Tidewater", "Vanguard", "Pathfinder",
    "Trident", "Blackstone", "Horizon", "Centennial", "Cobalt", "Cardinal",
    "Ironwood", "Starlight", "Timberline", "Meridian", "Pacifica", "Arrowhead",
    "Blue Ridge", "Cumberland", "Prairie Wind", "Maverick", "Redwood", "Thunderbird",
    "Orion", "Heritage", "Sentry", "Garrison", "Valor", "Bison", "Silverline",
    "Crestview", "TrueNorth", "IronHorse", "Frontline", "Vantage", "Endeavor",
    "Stratos", "Apex Point", "Summit Ridge", "Silver Creek", "Copperhead",
    "Wind River", "Blackhawk", "Redhawk", "Timberland", "Northline", "Highline",
    "Trans-Horizon", "Benchmark", "Iron Gate", "Steadfast", "Reliant"
]

NAME_MODIFIERS = [
    "", "National", "Regional", "Continental", "Overland", "Interstate",
    "Trans-American", "Express", "Direct", "Specialized", "Integrated",
    "Premier", "Allied", "Priority", "United", "Precision"
]

NAME_SUFFIXES = [
    "Logistics LLC", "Transport Inc.", "Freight Corp.", "Trucking Co.",
    "Carriers LLC", "Lines Inc.", "Express Logistics LLC", "Systems Inc.",
    "Haulers LLC", "Solutions Corp.", "Intermodal Services LLC",
    "Supply Chain Fleet LLC", "Dedicated Freight Lines Inc.", "Transit LLC",
    "Freightways Inc.", "Cartage Corp.", "Motor Freight LLC",
    "Transport Solutions Inc.", "Logistics Group LLC", "Expedited Services Inc."
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

def generate_random_carrier(i, used_names=None, used_dots=None, used_mcs=None):
    if used_names is None:
        used_names = set()
    if used_dots is None:
        used_dots = set()
    if used_mcs is None:
        used_mcs = set()

    while True:
        prefix = random.choice(NAME_PREFIXES)
        modifier = random.choice(NAME_MODIFIERS)
        suffix = random.choice(NAME_SUFFIXES)
        if modifier:
            full_base = f"{prefix} {modifier} {suffix}"
        else:
            full_base = f"{prefix} {suffix}"
        dba_opt = f" (DBA {prefix} Freight Solutions)" if random.random() < 0.20 else ""
        name = f"{full_base}{dba_opt}"
        if name not in used_names:
            used_names.add(name)
            break

    while True:
        dot = str(random.randint(1000000, 3999999))
        if dot not in used_dots:
            used_dots.add(dot)
            break

    while True:
        mc = str(random.randint(100000, 999999))
        if mc not in used_mcs:
            used_mcs.add(mc)
            break

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
    used_names = set()
    used_dots = set()
    used_mcs = set()
    carriers = [generate_random_carrier(i, used_names, used_dots, used_mcs) for i in range(500)]
    out_path = config.CARRIERS_JSON_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(carriers, f, indent=4)
    print(f"Generated 500 synthetic fictional carrier profiles at {out_path}")

if __name__ == "__main__":
    main()

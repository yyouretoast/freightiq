# DATA.md: FreightIQ Dataset Provenance & Schema Specification

## 1. Dataset Overview
FreightIQ operates on a curated, high-fidelity relational and vector dataset representing commercial motor carriers operating in interstate commerce across the United States.

- **Record Count:** 500 carrier profiles.
- **Relational Storage:** SQLite database (`data/carriers.db`) with WAL journal mode.
- **Lexical Inverted Index:** SQLite FTS5 virtual table (`carriers_fts`).
- **Dense Vector Embeddings:** ChromaDB Persistent Collection (`data/chroma_db`) embedded with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

---

## 2. Relational Schema (`carriers` table)

| Column Name | Type | Description / Constraints |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique internal database identifier |
| `carrier_name` | `TEXT NOT NULL` | Legal carrier name and DBA (e.g. *Apex Logistics LLC (DBA Apex Freight Solutions)*) |
| `dot_number` | `TEXT UNIQUE NOT NULL` | 7-digit USDOT identification number |
| `mc_number` | `TEXT UNIQUE NOT NULL` | 6-digit FMCSA Motor Carrier docket number |
| `hq_state` | `TEXT NOT NULL` | 2-letter US postal state code of carrier headquarters |
| `service_regions` | `TEXT NOT NULL` | JSON array of operational regions (`Midwest`, `Southeast`, `Northeast`, `Southwest`, `Mountain`, `Pacific Northwest`, `West Coast`) |
| `equipment_types` | `TEXT NOT NULL` | JSON array of active fleet trailer types (`dry van`, `flatbed`, `reefer`, `tanker`, `LTL`, `intermodal`) |
| `cargo_specializations`| `TEXT NOT NULL` | JSON array of primary cargo commodities (`general freight`, `fresh produce`, `pharmaceuticals`, `hazardous materials`, `electronics`, `building materials`, `machinery`, `automotive parts`, `retail goods`, `oversized loads`) |
| `safety_rating` | `TEXT NOT NULL` | FMCSA SAFER safety audit rating (`satisfactory`, `conditional`, `unsatisfactory`) |
| `years_operating` | `INTEGER NOT NULL` | Verified years of active interstate motor carrier operations (range: 2–35) |
| `contact_email` | `TEXT NOT NULL` | Dispatch contact email address |
| `notes` | `TEXT` | Rich narrative describing fleet capabilities, certifications (TWIC, GDP, Hazmat Class 3/8), equipment specifications, and telematics |

---

## 3. Data Generation & Ingestion Methodology
All database records are deterministically generated and seeded via:
```bash
python scripts/seed_db.py          # Seed if not already populated
python scripts/seed_db.py --force  # Force rebuild and re-index
```

### Architectural Details:
1. **Simulation & Synthetic Provenance Disclosure:** All carrier entity names, USDOT numbers, and compliance ratings are 100% fictional simulations generated for benchmark evaluation, ensuring zero misrepresentation of real motor carriers. Carrier profiles simulate real-world logistics nomenclature and FMCSA compliance structures without incorporating any proprietary or copyrighted corporate trademarks.
2. **Elevated Compliance Stress-Testing:** The synthetic corpus features an intentionally elevated risk distribution (60% satisfactory, 20% conditional, 20% unsatisfactory) compared to live commercial populations (<3% unsatisfactory) in order to rigorously stress-test FMCSA compliance gating and zero-row relaxation safety filters.
3. **Realistic Jargon & Specifications:** Carrier notes are built from authentic freight equipment terminology (e.g., *Carrier Vector multi-temp chillers capable of -20°F to 70°F*, *aluminum spread-axle flatbeds with 4-inch heavy-duty straps*, *DOT 407/412 sanitary stainless steel chemical tankers with rear pump-off*, *Moffett forklift offloading*, *port TWIC badges*).
4. **Idempotent Ingestion:** Both `rag/setup_sqlite.py` and `rag/ingest_chroma.py` inspect existing row counts against `len(carriers)`. Re-running `seed_db.py` does not duplicate records or cause primary key collisions.
5. **FTS5 Synchronization:** SQLite external content table `carriers_fts` mirrors text columns directly from the `carriers` table, providing sub-millisecond BM25 keyword matching with zero data redundancy.

---

## 4. Evaluation Corpus Scoping & Deployment Rationale
The decision to utilize a curated 500-profile synthetic evaluation corpus rather than a raw federal census dump was guided by operational and deployment requirements:
- **Fast Reproducibility & Cold-Start:** 500 richly structured carrier profiles seed and index in <20 seconds on standard CPU environments, enabling instant container startup on Hugging Face Spaces and fast CI/CD test runs.
- **Combinatorial Attribute Coverage:** The dataset provides complete combinatorial density across 20 high-volume freight corridor states, 7 operational regions, 6 trailer types, and 10 specialized commodity niches, enabling rigorous multi-constraint hybrid queries without sparse attribute drop-offs.
- **Hosting & Memory Constraints:** Lightweight embedded storage (ChromaDB + SQLite WAL) operates comfortably within standard free-tier container memory limits without requiring external database services or dedicated vector hosting.

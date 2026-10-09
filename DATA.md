# DATA.md: FreightIQ Dataset Provenance & Schema Specification

## 1. Dataset Overview
FreightIQ uses a relational and vector dataset representing 500 commercial motor carriers in interstate commerce across the United States.

- **Record Count:** 500 carrier profiles.
- **Relational Storage:** SQLite database (`data/carriers.db`) with WAL journal mode.
- **Lexical Index:** SQLite FTS5 virtual table (`carriers_fts`).
- **Dense Embeddings:** ChromaDB collection (`data/chroma_db`) embedded with `all-MiniLM-L6-v2` (384 dimensions).

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

### Implementation Details:
1. **Synthetic Data Disclosure:** Carrier entity names, USDOT numbers, and compliance ratings are synthetic simulations generated for evaluation. Profiles use realistic logistics nomenclature and FMCSA structures without incorporating proprietary corporate trademarks.
2. **Compliance Stress-Testing:** The synthetic corpus features an elevated risk distribution (60% satisfactory, 20% conditional, 20% unsatisfactory) compared to live commercial populations (<3% unsatisfactory) to test FMCSA compliance gating and zero-row relaxation safety filters.
3. **Domain Vocabulary:** Carrier notes include equipment specifications (e.g., *Carrier Vector multi-temp chillers (-20°F to 70°F)*, *aluminum spread-axle flatbeds with 4-inch heavy-duty straps*, *DOT 407/412 stainless chemical tankers*, *Moffett forklift offloading*, *port TWIC badges*).
4. **Idempotent Seeding:** `rag/setup_sqlite.py` and `rag/ingest_chroma.py` check existing row counts against `len(carriers)`. Re-running `seed_db.py` does not duplicate records.
5. **FTS5 Synchronization:** SQLite external content table `carriers_fts` mirrors text columns from the `carriers` table, enabling BM25 keyword matching without duplicating storage.

---

## 4. Evaluation Corpus Scoping & Deployment Rationale
A 500-profile synthetic evaluation corpus was chosen rather than a full census dump for operational and benchmark reasons:
- **Reproducibility & Startup:** 500 carrier profiles seed and index in <20 seconds on standard CPU environments, allowing fast container startup on Hugging Face Spaces and rapid CI/CD test runs.
- **Attribute Coverage:** The dataset provides complete combinatorial coverage across 20 freight corridor states, 7 operational regions, 6 trailer types, and 10 commodity categories, supporting multi-constraint hybrid queries without sparse attribute drop-offs.
- **Embedded Footprint:** Embedded storage (ChromaDB + SQLite WAL) operates within standard container memory limits without requiring external database instances or dedicated vector servers.

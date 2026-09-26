# ClinIQ — RAG-Powered AI for Substance Abuse Risk Detection

<p align="center">
  <img src="report/ClinIQ_LinkedIn.gif" alt="ClinIQ Demo" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Claude%20API-claude--haiku-orange?logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NSF%20NRT-4th%20Place%202026-gold" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **4th Place — NSF NRT Research-A-Thon 2026 · UMKC · Challenge 1 Track A: AI Modeling and Reasoning**

ClinIQ is a research prototype comparing keyword, embedding, and LLM+RAG methods on public drug-review data, with a separate synthetic-claims workflow. The reported corpus has 52,184 public review rows, not clinical EHR records or verified unique patients. Labels are keyword-derived proxies, not clinician-adjudicated SUD diagnoses. Review trends and mortality trends cannot establish individual linkage or causation.

**September 11 maintenance:** dashboard metrics now derive from saved confusion counts; unvalidated per-claim payment impact is unknown; scenario economics explicitly model eligibility, documentation, payment changes and costs. The original award/project scope is preserved. These repairs do not establish production clinical use or realized revenue.

Knowledge-base URLs provide provenance but do not, by themselves, validate every interpretation or coding recommendation.
---

## Why This Project Stands Out

| Signal | What ClinIQ demonstrates |
|--------|--------------------------|
| **RAG at scale** | pgvector IVFFlat index over 384-dim sentence embeddings; all-MiniLM-L6-v2 + Claude Haiku for multi-step reasoning |
| **3-method ML comparison** | Rule-based vs embedding cosine vs LLM+RAG — precision/recall/F1 against keyword proxies in a 600-record comparison |
| **Unsupervised discovery** | UMAP dimensionality reduction + HDBSCAN density clustering reveals clinically meaningful patient subpopulations |
| **Full-stack delivery** | PostgreSQL schema (11 tables), Streamlit dashboard, reproducible Docker setup, CI via GitHub Actions |
| **Domain exploration** | Connects research questions to a separate synthetic documentation-review workflow |
| **Ethical AI** | Population-level only, anonymized data, full source auditability, human-in-the-loop design |

---

## FHIR interoperability and healthcare data safety (September 26, 2026)

`interop/` ingests **FHIR R4 bundles** the way an EHR integration would, using the public
[Synthea](https://synthea.mitre.org/) synthetic-patient sample (no real patients, no PHI).

```
FHIR Bundle ─▶ validate ─▶ normalize ─▶ terminology map ─▶ de-identify ─▶ role-gated access ─▶ audit log
               (structure,   (Condition,   (SNOMED CT,        (keyed pseudonyms,  (analyst / researcher /  (hash-chained,
                references,   MedicationReq, RxNorm → ingredient, year-only dates,  admin; small cells       tamper-evident)
                code systems) Observation)   LOINC values)       90+ ages, no text)  suppressed <11)
```

**Run on the Synthea sample** (`python -m interop.evaluate`):

| | |
|---|---|
| Patients / bundles | 109 / 109 |
| FHIR resources processed | 118,868 in 3.84 s |
| Accepted (Condition / MedicationRequest / Observation) | 3,540 / 3,858 / 46,214 |
| Quarantined with a reason | 91 (Observations coded in SNOMED CT instead of LOINC) |
| PHI leaks found by the scanner | 0 (names, phones, addresses, ZIPs, SSN/MRN identifiers, birth dates) |
| Audit chain | valid; analyst request for row-level records denied and logged |

Medication orders that point to a separate `Medication` resource (`medicationReference`) are resolved,
and RxNorm display strings are reduced to ingredients (`Abuse-Deterrent 12 HR Oxycodone Hydrochloride 15 MG ...`
→ `oxycodone hydrochloride`).

**Why cohorts are defined by codes, not embeddings.** Each patient's de-identified summary was indexed for
free-text search, and each cohort was retrieved at k = true cohort size:

| Cohort | Size | BM25 recall | Dense recall | Code filter |
|---|---|---|---|---|
| patients with diabetes | 8 | 0.62 | 0.25 | 1.00 |
| patients with prediabetes | 34 | 0.82 | 0.618 | 1.00 |
| patients with hypertension | 20 | 0.80 | 0.55 | 1.00 |
| patients with obesity | 44 | 0.93 | 0.591 | 1.00 |
| patients with anemia | 35 | 0.89 | 0.514 | 1.00 |
| patients with chronic pain | 27 | 0.85 | 0.63 | 1.00 |
| patients with ischemic heart disease | 13 | 1.00 | 0.846 | 1.00 |
| patients with chronic kidney disease | 6 | 1.00 | 0.5 | 1.00 |
| patients with a substance use problem | 9 | 0.11 | 0.222 | 1.00 |
| patients taking an opioid | 6 | 0.17 | 0.5 | 1.00 |
| patients taking metformin | 4 | 1.00 | 0.5 | 1.00 |
| patients taking a statin | 18 | 0.33 | 0.5 | 1.00 |

Mean recall: BM25 0.711, dense 0.518. Semantic search misses
a large share of each code-defined cohort (the substance-use cohort worst of all), so cohort selection uses
terminology-mapped filters and embeddings are kept for exploration and summarization.

Tests: `tests/test_fhir_interop.py` (ingestion, all three terminologies, de-identification of every identifier,
leak scanner, dangling references, RBAC suppression, audit tampering detection) run in CI with no downloads.

## Architecture

```
Public Review Rows (52,184)
        │
        ▼
┌───────────────────────────────────────────────────┐
│              Signal Detection Layer                │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │  Rule-Based  │  │  Embedding   │  │LLM+RAG  │  │
│  │ F1=0.854     │  │ Recall=1.000 │  │Prec=0.94│  │
│  └──────────────┘  └──────────────┘  └─────────┘  │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│           Temporal + Behavioral Analysis           │
│  Year-over-year trends  │  UMAP + HDBSCAN clusters │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│           Clinical Bridge (RAG Agent)              │
│  ICD-10 gap detection  │  DRG revenue impact       │
│  pgvector retrieval    │  Claude Haiku reasoning    │
└───────────────────────────────────────────────────┘
        │
        ▼
     Streamlit Dashboard (5 panels, live RAG)
```

---

## Key Results

### Detection Performance (600-record balanced proxy-label comparison)

| Method | Precision | Recall | F1 | Best Use Case |
|--------|-----------|--------|----|---------------|
| Rule-Based (ICD-10 vocab) | 0.861 | 0.847 | **0.854** | Keyword-proxy baseline |
| Embedding (cosine ≥ 0.32) | 0.504 | **1.000** | 0.670 | 100% proxy recall in this sample; 295 false positives |
| LLM + RAG (Claude Haiku) | **0.938** | 0.400 | 0.561 | Highest proxy precision in this saved sample |

### Temporal Findings (2008–2017 opioid crisis arc)

- **3× volume surge**: SUD review volume from 2014→2016 tracks the CDC-documented fentanyl influx
- **18× distress escalation**: Patient distress proportion rose from 1.7% (2008) to 30.5% (2017)
- **Composition shift**: Opioid-specific proportion fell 40%→16% while total distress rose — the crisis diversified beyond opioids

### Financial interpretation

The earlier $27.5M–$60M range multiplied 10,000 by an assumed $2,750–$6,000. It was a scenario, not measured recoverable or collected revenue. A suggested diagnosis does not automatically change DRG payment. Clinical documentation, coding eligibility, the claim's existing grouping, payer contract, realization and review costs all matter. Hospital payment increases are not automatically payer savings.

The dashboard now separates historical synthetic dollar fields from a transparent hypothetical funnel. Example: 10,000 admissions × 5% candidates × 40% missed × 60% valid × 50% payment-changing × 100% realization × $3,000 = $180,000 gross. Review of 500 candidates at $30 costs $15,000: $165,000 before implementation and other costs. No part is measured project revenue.

`analysis/evidence.py` recomputes P/R/F1 from saved counts. `claim_payment_delta` returns unknown without validated before/after payments. The agent also returns unknown instead of assigning automatic dollars per suggested code. Public reviews and synthetic claims remain separate datasets.

Saved confusion counts: Rule TP=254, FP=41, FN=46, TN=259; embedding TP=300, FP=295, FN=0, TN=5; LLM+RAG TP=120, FP=8, FN=180, TN=292. See `outputs/method_comparison_results.csv`. The 600-case model run was not repeated during maintenance; these metrics were recomputed from its saved counts.

Run `python -m pytest tests/ -q` for offline tests of the shared application baseline, undefined metrics, saved confusion counts, and the financial funnel. Live database/LLM integration and clinical adjudication remain separate validation work.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Anthropic Claude Haiku (via API) |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim, sentence-transformers) |
| **Vector DB** | PostgreSQL 16 + pgvector (IVFFlat index, cosine distance) |
| **Clustering** | UMAP + HDBSCAN |
| **Data** | pandas, scikit-learn, scipy |
| **Visualization** | Plotly, Streamlit |
| **Infrastructure** | Docker Compose, GitHub Actions CI |
| **Data Sources** | CDC API, CMS IPPS FY2024, ICD-10-CM, NIDA, Kaggle |

---

## Quick Start

### Option A — Docker (Recommended, zero configuration)

```bash
git clone https://github.com/vajja1405/ClinIQ-AI-for-Substance-Abuse-Risk-Detection.git
cd ClinIQ-AI-for-Substance-Abuse-Risk-Detection
cp .env.template .env          # add your ANTHROPIC_API_KEY
docker compose up --build      # starts PostgreSQL + pgvector + Streamlit app
```

Open `http://localhost:8501`

### Option B — Local

```bash
git clone https://github.com/vajja1405/ClinIQ-AI-for-Substance-Abuse-Risk-Detection.git
cd ClinIQ-AI-for-Substance-Abuse-Risk-Detection
make setup                     # creates venv + installs deps + copies .env
# Edit .env: set DATABASE_URL and ANTHROPIC_API_KEY
make db-docker                 # starts PostgreSQL with pgvector via Docker
make db-init                   # loads schema, data, and builds RAG index
make pipeline                  # runs full analysis (Task 1 + Task 2 + Agent)
make dashboard                 # launches Streamlit on http://localhost:8501
```

### Environment Variables

```
DATABASE_URL=postgresql://cliniq:cliniq@localhost:5432/cliniq
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project Structure

```
cliniq/
├── data/
│   ├── load_reviews.py              # Load & classify 52,184 Kaggle drug reviews
│   └── load_public_health_data.py   # Download CDC, CMS, ICD-10 sources
│
├── db/
│   ├── schema.sql                   # 11-table PostgreSQL schema + pgvector
│   └── setup_db.py                  # Database initialization
│
├── agent/
│   ├── build_rag.py                 # Embed government docs into pgvector
│   ├── cliniq_agent.py              # Clinical gap detector (RAG + Claude)
│   └── explainability.py            # XAI justification generator
│
├── analysis/
│   ├── task1_signal_detection.py    # 3-method SUD detection + evaluation
│   └── task2_temporal_behavioral.py # Temporal trends + UMAP/HDBSCAN
│
├── streamlit_app/
│   └── app.py                       # Interactive 5-panel dashboard
│
├── tests/
│   └── test_detection.py            # Unit tests (no DB/API required)
│
├── raw_sources/                     # Auditable government documents
├── outputs/                         # Generated CSVs and JSON results
├── docker-compose.yml               # PostgreSQL + app in one command
├── Dockerfile
├── Makefile                         # make setup / run / test / dashboard
├── run_pipeline.py                  # Master orchestration script
└── requirements.txt                 # Pinned dependencies
```

---

## Database Schema

```sql
drug_reviews        -- 52,184 public reviews with keyword-proxy labels + signal category
cdc_overdose        -- CDC mortality data by year/state/substance
rag_embeddings      -- pgvector knowledge base (384-dim, IVFFlat, cosine)
rag_source_registry -- Audit trail: every source URL + document registered
dim_diagnosis       -- ICD-10-CM codes; claim-specific CC/MCC eligibility requires verification
dim_patient         -- Synthetic patient demographics
dim_provider        -- Hospital department/specialty lookup
fact_claims         -- Synthetic clinical claims
ai_risk_findings    -- Synthetic claim suggestions + unvalidated legacy scenario fields
method_comparison   -- Task 1 precision/recall/F1 by method
temporal_analysis   -- Task 2 year-level SUD volume + distress metrics
```

---

## Data Sources

All knowledge base content sourced from real, publicly available government documents. Every `rag_embeddings` row carries `source_url` + `source_document`.

| Source | What it provides |
|--------|-----------------|
| Kaggle UCI Drug Reviews | 215k patient reviews, 2008–2017 |
| CDC Drug Overdose Surveillance API | Annual overdose death rates by state |
| ICD-10-CM 2024 (CDC FTP) | Official SUD codes F10–F19, T40, T43 with CC/MCC status |
| CMS IPPS FY2024 Final Rule | DRG relative weights and payment rates |
| CMS ICD-10-CM Coding Guidelines | Official Section I.C.5 substance use rules |
| NIDA Trends & Statistics | Population-level SUD prevalence data |

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

Tests are self-contained — no database or API key needed.

---

## Ethical AI Principles

- **No individual identification** — all reviews are anonymized; clinical data is synthetic
- **Population-level only** — every analysis is aggregated across cohorts
- **Full auditability** — every knowledge base chunk traces to a government URL
- **Human-in-the-loop** — all AI findings require analyst review before action
- **Public data only** — CDC, CMS, ICD-10-CM; no proprietary clinical records

---

## Team

| Name | Role | Email |
|------|------|-------|
| Rahul Vajja | RAG pipeline, clinical bridge, database, dashboard | rcvtk3@umsystem.edu |
| Bhavani Adula | Temporal analysis, behavioral clustering, ethics framework | barh3@umsystem.edu |

**Faculty Advisors:** Dr. Mostafizur Rahman · Dr. Yugyung Lee  
**Program:** NSF NRT — AI for Health Informatics, UMKC

---

## Citation

```bibtex
@misc{cliniq2026,
  title   = {ClinIQ: RAG-Powered AI for Substance Abuse Risk Detection from Social Signals},
  author  = {Vajja, Rahul and Adula, Bhavani},
  year    = {2026},
  note    = {NSF NRT Research-A-Thon 2026, UMKC — 4th Place, Challenge 1 Track A},
}
```


## Plan reviewer workload before a pilot

The Method Comparison panel now estimates review volume, missed signals, false
alerts and review hours from the saved confusion counts and explicit prevalence/time
assumptions. Download the scenario as CSV. Proxy-label agreement does not establish
clinical accuracy or transfer to a new population. Saved panels no longer require
constructing an Anthropic client at startup. No new clinical evaluation is claimed.

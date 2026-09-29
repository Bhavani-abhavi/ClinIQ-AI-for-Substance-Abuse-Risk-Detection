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

## Synthetic prior-authorization agent (September 27, 2026)

`prior_auth/` is a **LangGraph** workflow that reviews prior-authorization requests against the FHIR
records above. Policies and patients are both synthetic: the four policies were written for this project
in the style of common payer criteria and are **not any payer's actual policy**. This is decision support
for a clinical reviewer, not a coverage system.

```
intake ─▶ evaluate ─▶ write_draft ─▶ check ─┬─ REVISE (≤2) ─▶ write_draft
          (RBAC-checked   (local LLM)   (deterministic)  └─▶ decide ─┬─ APPROVE ─▶ finalize (audit, ClaimResponse)
           record fetch +                                            └─ PEND ─▶ clinician_review [interrupt] ─▶ finalize
           criteria engine)
```

- **Criteria engine** (`engine.py`): each criterion is data (diagnosis, lab threshold with look-back,
  step therapy, duration, BMI with comorbidity, age) and returns *met*, *not met* or *missing* with the
  FHIR resources it relied on. *Missing* is a documentation request, not a denial.
- **Only a clinician can deny.** The agent may auto-approve when every criterion is met; everything else
  pauses at a `clinician_review` interrupt and resumes from its checkpoint. A denial from any role other
  than `clinician` is refused and returned to the queue.
- **The model writes, the engine decides.** The LLM drafts the reviewer summary. A deterministic checker
  rejects drafts whose statuses, citations or numbers disagree with the engine, or that use denial
  language; after two failed revisions the summary falls back to the engine's own wording.
- **Minimum necessary access.** The agent runs as a `utilization_review` role that can read one patient's
  de-identified record per request, and every read and decision lands in the hash-chained audit log.
  Output is shaped after a FHIR R4 `ClaimResponse` (Da Vinci PAS style; not validated against the IG).

**Run on the Synthea sample** (`python -m prior_auth.evaluate --model`):

| | GLP-1 (T2D) | PCSK9 (lipids) | Lumbar MRI | Bariatric | Total |
|---|---|---|---|---|---|
| Requests (patients with a related problem) | 37 | 18 | 18 | 45 | 118 |
| Auto-approved (all criteria met) | 1 | 13 | 15 | 2 | 31 |
| Pended for a clinician | 36 | 5 | 3 | 43 | 87 |

- **Auto-denials: 0.** Every pended case resumed from its checkpoint; the audit chain verified (236 entries).
- **Missing-documentation test:** for each approved case, the evidence behind one criterion at a time was
  deleted from the record. All **89/89** variants pended and named exactly the criteria that relied on
  the deleted evidence.
- **Drafting with a local model (Llama 3.2 3B via Ollama, CPU):** of 40 reviewer summaries, 30 passed the
  checker on the first draft, 2 after one revision, and 8 fell back to the engine's own wording; no draft
  that failed a check reached a reviewer. Median 42 s per case, about 540 prompt tokens. A first run
  scored 0/40 because of a checker bug, not the model: citations were compared verbatim (the model wrote
  `Condition/…: Chronic low back pain (2014)`) and the CPT code and review year shown in the prompt were
  not allowed as numbers. The checker now extracts ref ids and allows numbers from the prompt; a test
  covers it.

Tests: `tests/test_prior_auth.py` (engine statuses, auto-approval and audit, clinician-only denial, checker
catches for status, citation, number and denial-language errors, revision and template fallback).

## Labeling workbench with active learning (September 28, 2026)

Proxy labels made every result above possible, and every result above is limited by them. `labeling/` is the tool for replacing them with human labels at the lowest labeling cost.

- **Active-learning queue.** A TF-IDF + logistic regression model (review text only) retrains every 25 labels and serves the review it is least sure about next. Its guess is shown as a pre-label. Until it has seen both classes, the queue alternates between the review with the most substance-use terms and a random-order review, because only about 6% of reviews are relevant.
- **Quality control.** Hidden gold items (one task in ten) score each annotator. About 10% of items go to a second annotator, which gives Cohen's kappa. Disagreements wait in a conflict queue until someone adjudicates them. Labels faster than 2 seconds are flagged, and annotators below 80% gold accuracy are flagged.
- **Data checks on import.** Text is HTML-unescaped and whitespace-normalized; reviews under 10 words, exact duplicates and rows without an id are rejected and counted.
- **UI.** React + TypeScript (Vite), keyboard shortcuts (`1`, `0`, `S`), highlighted substance-use terms, and a quality view with annotator scores, agreement, conflicts, and model average precision after each retrain. Resolved labels export as JSONL.

**How many labels does active learning save?** `python -m labeling.active_learning --csv ...` simulates annotators with the proxy labels: every strategy starts from the same 100 random labels, asks for 100 at a time, retrains, and is scored on the same 600-review test set as DistilBERT. Averaged over 10 seeds:

| Labels | Random: average precision | Uncertainty: average precision | Random: relevant reviews found | Uncertainty: relevant reviews found |
|---|---|---|---|---|
| 500 | 0.718 | 0.784 | 29 | 203 |
| 1,000 | 0.754 | 0.848 | 58 | 449 |
| 3,000 | 0.812 | 0.910 | 172 | 1,046 |

Trained on all 46,307 pool labels, the model reaches average precision 0.917. Uncertainty sampling gets to 95% of that (0.872) with **1,500 labels**; random sampling needs **11,000**, 7.3 times as many. Average precision is used because the pool is about 6% positive and the test set is 50% positive: small models put almost every test review below 0.5, so F1 at a fixed threshold would mostly measure that prior shift. The proxy labels stand in for annotators, so no human labeling time is measured. Results: `outputs/active_learning.json`.

Run it:

```bash
python -m labeling.server --demo                                  # synthetic reviews, no dataset needed
python -m labeling.server --csv /path/to/drugsComTest_raw.csv     # 3,000 real reviews + 60 gold items
cd labeling/ui && npm install && npm run build                    # UI served at http://127.0.0.1:8765
```

Tests: `tests/test_labeling_workbench.py` (queue order, gold scoring, agreement, adjudication, export, API), `labeling/ui/src/logic.test.ts` (Vitest), and `labeling/ui/e2e/workbench.spec.ts`, a Playwright flow run on Chromium, Firefox and WebKit, each against its own demo server. Two annotators label with the keyboard and mouse, disagree, and a reviewer resolves the conflict; a second test checks a 375-pixel-wide phone layout. CI runs all three browsers.

**Reliability.**
- **Saves:** a retried save with the same label is a no-op, so a network retry never double-counts. A different
  label, or a second reviewer's conflicting adjudication, returns HTTP 409 with what is stored. Changing a label
  on purpose is an explicit revision, kept in `label_events`.
- **UI:** unsaved labels go into an on-device outbox. Network errors and 5xx responses are retried with backoff,
  and anything left over is sent after a reload. `labeling/ui/e2e/reliability.spec.ts` covers this on all three
  browsers:
  - the connection drops after the server stored the label (stored once);
  - the tab goes offline and reloads (sent once after reload);
  - the server returns 503 past the automatic retries (manual retry succeeds).
- **Large queue:** `python -m labeling.benchmark` loads all 46,296 pool reviews into a SQLite
  file and labels 600. At the 95th percentile:

  | Operation | Time |
  |---|---:|
  | Fetch the next task | 12.9 ms |
  | Save a label | 0.54 ms |
  | Save that triggers a retrain | 122.15 ms |
  | Import and first fit (one-time) | 7.57 s |

  Results: `outputs/labeling_benchmark.json`.

### Counterfactual bias test (September 28, 2026)

Whether a review is about substance use should not depend on who wrote it. `python -m analysis.bert_bias` rewrites each of the 600 test reviews two ways and measures how often the text-only DistilBERT changes its answer: with gender words swapped, and with an identity statement in front ("As a Black woman, ..."). Neutral prefixes ("As a person, ...") are the control. A prefix pushes the end of long reviews past the 128-token limit (266 of the 600 are longer), and the control measures that effect alone. The pass criterion, fixed before running, is at most 2% of predictions flipping for every perturbation.

| Perturbation | Original model | After counterfactual augmentation |
|---|---|---|
| Gender words swapped | 1.0% | 0.0% |
| Neutral prefix (control: person / patient) | 2.8% / 2.2% | 1.7% / 2.3% |
| Worst identity statement | 9.0% ("gay man") | 3.0% ("person on disability") |
| "Gay man" | 9.0% | 2.2% |
| Identities above the control's flip rate by more than 1 point | 8 of 12 | 0 of 12 |
| F1 on the 600 test reviews | 0.873 | 0.871 |

The original model failed: identity statements flipped 2.5–9.0% of predictions, mostly toward "not relevant" (196 of 318 flips; 51 of 54 for "gay man"). The fix is counterfactual data augmentation: `python -m analysis.bert_classifier --text-only --augment-identity` rewrites half the training reviews with an identity statement and, half of those times, swapped gender words, leaving the labels unchanged. The training identities are a different list from the tested ones, so the test measures generalization, not memorized phrases. After retraining, every tested identity is within 0.7 points of the neutral control and F1 is similar (0.873 to 0.871). **The model still fails the 2% criterion**: four identities flip 2.2–3.0%, about the same as truncation alone. Handling long reviews (a longer context, or head-and-tail truncation) is the next fix. The gender swap is word-level and approximate ("her" always becomes "his"). The reviews carry no demographic fields, so this measures sensitivity to identity words, not outcome parity between real groups.

Results: `outputs/bert_bias.json` (original) and `outputs/bert_bias_cda.json` (augmented), with the augmented model's accuracy in `outputs/bert_eval_textonly_cda.json` (W&B run `54liflfc`). Tests: `tests/test_bert_bias.py`.

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

### Fine-tuned DistilBERT, tracked in Weights & Biases (September 28, 2026)

`analysis/bert_classifier.py` fine-tunes `distilbert-base-uncased` with the Hugging Face Trainer (2 epochs, CPU) and logs each run to Weights & Biases, offline by default.

- **Labels** are the project's keyword proxy (the review's condition, falling back to the drug name), not clinician adjudication.
- **Test set:** the original recipe, rebuilt: 300 SUD-relevant reviews (the 60 most useful per signal category) plus 300 non-SUD reviews drawn with a fixed seed. Test review texts are removed from training, because the dataset repeats reviews under brand and generic names. Training uses 7,366 reviews (positives plus twice as many negatives) and 818 for validation.
- **Rules** are the patient-voice keyword dictionary re-run on the same 600. The original benchmark also added ICD-10 terms from the database, which isn't available offline, so the rules score 0.841 here instead of 0.854.

| Inputs | Method | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Review text only | Fine-tuned DistilBERT | 0.910 | 0.840 | **0.873** | 0.950 |
| Review text only | Keyword rules | 0.871 | 0.677 | 0.762 | – |
| Drug name + review | Fine-tuned DistilBERT | 1.000 | 0.873 | 0.932 | 0.990 |
| Drug name + review | Keyword rules | 0.886 | 0.800 | 0.841 | – |

**Read the text-only rows first.** The drug name can set the proxy label by itself (a Suboxone review is labeled SUD-relevant whatever it says), so the drug-name rows partly measure who learns drug names. On review text alone, the fine-tuned model finds 49 more of the 300 SUD-relevant reviews than the rules, with 5 fewer false positives.

**Limits.** The test positives are the most-upvoted reviews in each category, which tend to be explicit. On the random validation split, text-only F1 is 0.755, so expect lower numbers on ordinary reviews. Both labels and test set are proxies; clinician-labeled data would be needed before any clinical use.

Reproduce (about 30 minutes per run on an Apple M3 CPU; `pip install -r requirements-bert.txt`):

```bash
python -m analysis.bert_classifier --csv /path/to/drugsComTest_raw.csv                      # drug name + review
python -m analysis.bert_classifier --csv /path/to/drugsComTest_raw.csv --text-only --out outputs/bert_eval_textonly.json
python -m analysis.bert_classifier --csv /path/to/drugsComTest_raw.csv --rules-only --out outputs/bert_eval.json  # rule baselines only
wandb sync outputs/wandb/offline-run-*                                                      # after `wandb login`
```

Results are in `outputs/bert_eval.json` and `outputs/bert_eval_textonly.json` (W&B runs `w3fu0hzl` and `nkcre65v`, project `cliniq-sud-detection`). Model weights (`models/bert-sud*/`) and W&B run files (`outputs/wandb/`) are not committed. `tests/test_bert_splits.py` checks the split, label and baseline logic without downloading a model.

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
| **Fine-tuning** | DistilBERT (Hugging Face Transformers + Trainer, PyTorch), Weights & Biases tracking |
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

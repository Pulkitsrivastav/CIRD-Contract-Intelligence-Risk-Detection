# Contract Intelligence & Risk Detection System (CIRD)

CIRD is an academic Streamlit application for preliminary screening of **Mutual Non-Disclosure Agreements (NDAs)** in PDF format. It extracts contract text locally, checks ten clause categories against a local reference baseline, identifies omissions and configured risk indicators, calculates a transparent 0–100 screening score, and exports Markdown and JSON reports.

> **Offline-only design:** Contract analysis uses the built-in local contextual NLP/rule engine plus a bundled trained TF-IDF clause classifier. The application has no online analysis mode, does not call an external LLM/API, and does not require an API key or model credentials.
>
> **Academic screening disclaimer:** Risk scores are indicators based on the configured educational baseline. This tool is not legal advice and does not replace review by a qualified lawyer.

---

## 1. Project structure

```text
Contract_Intelligence_Risk_Detection/
├── app.py                        # Streamlit interface and dashboard
├── analyzer.py                   # Offline analysis orchestration, risk scoring and reports
├── offline_engine.py              # Context-aware local clause and red-flag detection
├── fuzzy_logic.py                 # Mamdani fuzzy risk inference
├── csp_solver.py                  # Constraint consistency checks (AC-3/backtracking)
├── state_space.py                 # BFS workflow-state path (offline-only)
├── knowledge_graph.py             # NDA clause knowledge graph
├── document_utils.py              # Local PDF extraction and report saving
├── review.py / review_store.py    # Human-review and supplemental-baseline workflow
├── evaluation.py                  # Bundled-sample regression evaluation
├── data/nda_standards.json        # Ten standard NDA clause categories (A–J)
├── data/human_approval_store.json # Local human approval examples and pending queue
├── tests/                         # Automated regression tests
└── sample_documents/              # Standard and intentionally flawed sample NDAs
```

## 2. Setup and run (Windows / VS Code)

Open the extracted project folder in VS Code and run these commands in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown by Streamlit (normally `http://localhost:8501`). No `.env` file, API key, external model, or online account is needed for contract analysis. An internet connection is needed only if you choose to install the Python dependencies or access the application from a separately hosted server.

Optional: `CIRD_APPROVAL_STORE_PATH` can point the human-approval store to a persistent mounted volume when deploying the app. It is unrelated to analysis and is not an API credential.

## 3. Offline analysis workflow

1. **PDF validation and extraction:** PyMuPDF extracts readable text page by page. Empty, encrypted, scanned/image-only or otherwise unreadable PDFs are flagged.
2. **Local contextual detection:** `offline_engine.py` uses clause-specific concept groups, phrase patterns, carefully guarded single-term patterns, sentence context, nearby sections/headings and negation-aware checks. It also looks for configured red-flag wording such as perpetual confidentiality and uncapped liability.
3. **Reference comparison:** The detected evidence is compared with the ten-category local NDA baseline (A–J). Each category is classified as Present, Needs Review, or Missing / Not Detected.
4. **Fuzzy risk scoring:** Mamdani fuzzy inference combines local concept coverage and red-flag strength; the clause importance weights contribute to the overall 0–100 screening score.
5. **CSP consistency checks:** The constraint solver checks consistency between clause status and risk labels.
6. **Human review:** Novel or higher-risk provisions can be permanently approved as supplemental examples, semi-approved into a pending queue, or rejected/removed. The core reference baseline is not overwritten.
7. **Reports:** The dashboard exports Markdown and JSON results.

### Ten standard clause categories

- **A.** Definition of Confidential Information
- **B.** Permitted Use and Purpose
- **C.** Exclusions from Confidentiality
- **D.** Confidentiality Period
- **E.** Return or Destruction of Information
- **F.** Permitted Disclosure / Legal Compulsion
- **G.** Security and Safeguarding
- **H.** Liability and Remedies
- **I.** Governing Law and Jurisdiction
- **J.** Termination

## 4. Demonstration scenarios

Use the sample buttons in the sidebar or upload the PDFs directly:

- `sample_documents/standard_mutual_nda.pdf`: expected to detect all ten categories as Present and return a low-risk result.
- `sample_documents/high_risk_flawed_nda.pdf`: intentionally omits several categories and includes perpetual confidentiality / unlimited liability wording; expected to identify the missing categories and severe risk indicators.
- Upload a blank, scanned or unreadable PDF to test extraction validation.

These are deterministic functional regression examples, not a statistically representative accuracy benchmark.

## 5. Testing

```powershell
python -m unittest discover -s tests -v
```

The suite checks the offline clause engine, sample documents, evidence extraction, fuzzy scoring, CSP consistency, knowledge graph, PDF handling, human-approval workflow and report generation.

## 6. Important limitations

- **Deterministic screening, not a trained language model:** The offline engine is a locally executed, context-aware rule/pattern-based NLP system. It does not contain a pretrained Transformer or a neural model trained on a large contract corpus.
- **Context and wording:** Pattern coverage is broader than simple exact-keyword matching, but unusual drafting, complex legal dependencies and novel paraphrases can still be missed or misclassified.
- **Text-based PDFs:** Scanned image-only PDFs require OCR, which is not included.
- **NDA scope:** The supplied reference library is for Mutual NDAs. Other agreement types need their own baselines and validation.
- **Legal oversight:** Always inspect the cited source text and consult a qualified legal professional before relying on a contract decision.


## Offline learned clause classifier

CIRD now combines its explainable clause-specific rules with a locally trained TF-IDF classifier using word and character n-grams. The bundled model is stored at `data/offline_clause_classifier.joblib`; its labelled training examples are in `data/clause_training_examples.jsonl`. The training examples cover the ten configured NDA clause families plus unrelated text. They are manually authored educational examples, not a representative legal corpus. The model abstains on uncertain predictions and an ML-only match is sent to **Needs Review**; it is never auto-approved.

To retrain locally after adding reviewed examples:

```bash
pip install -r requirements.txt
python train_offline_model.py
```

No API key or network connection is used. The primary risk score is now calculated from the final clause risk labels (importance-weighted Low=10, Medium=50, High=90). The fuzzy score remains a separate diagnostic and cannot contradict the headline risk category.

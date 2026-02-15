# MLOps Hands-On Tutorial: Feast + MLflow

**Goal:** Build an end-to-end local ML pipeline to learn **feature management** (Feast) and **model lifecycle management** (MLflow) — the two systems most central to production ML workflows.

**Task:** Classify activity type (walking, running, resting, etc.) from wearable sensor data using LightGBM  
**Dataset:** [Apple Watch and Fitbit Data (Kaggle)](https://www.kaggle.com/datasets/aleespinosa/apple-watch-and-fitbit-data/data)

---

## Setup

**Prerequisites:** Python 3.9+, Kaggle account

```bash
# Create and activate a virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install feast mlflow lightgbm pandas scikit-learn
```

**Project structure:**

```
ml_ops_playground/
├── data/                    # Raw Kaggle dataset
├── notebooks/               # Exploration and experimentation
├── feature_repo/            # Feast feature definitions
│   ├── feature_store.yaml
│   └── features.py
├── src/
│   ├── baseline.py          # Phase 1A baseline model
│   ├── feast_training.py    # Phase 1B feature-store-backed training
│   ├── experiments.py       # Phase 2 MLflow experiment runs
│   └── serve_predict.py     # Phase 3 inference script
└── README.md
```

---

## Phase 1: Baseline → Feature Store

**Time estimate:** 3–4 hours

### Part A — Baseline Model

1. **Download and explore the dataset.** Load the CSV into pandas. Inspect columns (`heart_rate`, `steps`, `calories`, `acceleration_x/y/z`, `activity`, etc.), check for nulls, look at class distributions across activity types.

2. **Train a baseline LightGBM classifier** on the raw features with minimal preprocessing (label-encode the target, drop ID/timestamp columns, train/test split). Use default hyperparameters.

3. **Evaluate and save results.** Record accuracy, macro F1, and a confusion matrix. These are your "before" numbers to compare against the feature-store-backed model.

### Part B — Feature Store with Feast

4. **Initialize Feast.** From your project root:
   ```bash
   cd feature_repo
   feast init fitness_features
   ```
   Configure `feature_store.yaml` to use a local SQLite offline store and file-based online store.

5. **Define feature views.** Create two feature views in `features.py`:
   - **Rolling heart rate statistics:** 5-minute and 15-minute rolling mean and standard deviation of heart rate, grouped by participant.
   - **Cross-feature interaction:** Acceleration magnitude (`sqrt(x² + y² + z²)`) multiplied by heart rate — a proxy for exertion intensity.

   Define an `Entity` for the participant and set appropriate TTLs (time-to-live).

6. **Materialize features.**
   ```bash
   feast apply
   feast materialize <START_TIMESTAMP> <END_TIMESTAMP>
   ```

7. **Build a training dataset with point-in-time correctness.** Use `get_historical_features()` to join your entity dataframe (participant ID + event timestamp + label) against the feature views. This guarantees no future data leaks into training rows.

8. **Retrain LightGBM** on the Feast-backed feature set and compare metrics against the baseline. The improvement (or lack thereof) is secondary — the point is experiencing the workflow.

### Key concepts practiced

- Feature definitions as version-controlled code
- Point-in-time joins and why they prevent training/serving skew
- Materialization: populating offline and online stores from raw data

---

## Phase 2: Experiment Tracking & Model Registry

**Time estimate:** 3–4 hours

### Experiment Tracking

1. **Start the MLflow tracking server:**
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
   Set the tracking URI in your scripts:
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://127.0.0.1:5000")
   mlflow.set_experiment("activity-classifier")
   ```

2. **Run three logged experiments:**

   | Run | Description |
   |-----|-------------|
   | 1 | Baseline: raw features, default hyperparameters |
   | 2 | Feast features, default hyperparameters |
   | 3 | Feast features, tuned hyperparameters (grid or random search over `num_leaves`, `learning_rate`, `n_estimators`) |

3. **For each run, log:**
   - **Parameters:** feature set name, all LightGBM hyperparameters
   - **Metrics:** accuracy, macro F1, per-class F1
   - **Artifacts:** confusion matrix plot (saved as PNG), feature importance plot
   - **Tags:** `feature_set: "raw"` or `"feast_v1"`, a short description of the run

4. **Compare runs in the MLflow UI** at `http://127.0.0.1:5000`. Sort by F1, visually inspect confusion matrices, confirm that logging is working as expected.

### Model Registry & Lifecycle

5. **Register the best model.** From the MLflow UI or programmatically:
   ```python
   mlflow.register_model(f"runs:/{best_run_id}/model", "activity_classifier")
   ```
   This creates version 1 in stage `None`.

6. **Transition v1 through the lifecycle:**
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient()

   # Promote to Staging
   client.transition_model_version_stage("activity_classifier", 1, "Staging")

   # Promote to Production
   client.transition_model_version_stage("activity_classifier", 1, "Production")
   ```

7. **Create v2.** Make a deliberate change (e.g., drop the cross-feature interaction, or add a new feature), retrain, log the run, and register as a new version. Promote v2 to Production and archive v1:
   ```python
   client.transition_model_version_stage("activity_classifier", 2, "Production")
   client.transition_model_version_stage("activity_classifier", 1, "Archived")
   ```

### Key concepts practiced

- Systematic experiment comparison with full provenance
- Model versioning tied to specific runs and artifacts
- Lifecycle management: `None → Staging → Production → Archived`

---

## Phase 3: Local Serving Integration

**Time estimate:** 1–2 hours

1. **Serve the Production model:**
   ```bash
   mlflow models serve -m "models:/activity_classifier/Production" -p 1234 --no-conda
   ```

2. **Write an inference script** (`src/serve_predict.py`) that simulates the production integration pattern:
   - Accept a participant ID and timestamp as input
   - Fetch features from the **Feast online store** using `get_online_features()`
   - Format features into the model's expected input schema
   - Send a POST request to the MLflow serving endpoint at `http://127.0.0.1:1234/invocations`
   - Return the predicted activity type

   This script is the key artifact of the tutorial — it demonstrates how a feature store and model server connect at inference time.

3. **Test with a few sample requests** and verify predictions are reasonable.

### Key concepts practiced

- Online feature retrieval vs. offline (training-time) retrieval
- Model serving via REST endpoint
- The full loop: features defined once, used consistently in training and serving

---

## Timeline Summary

| Phase | Time | Priority |
|-------|------|----------|
| Phase 1: Baseline → Feature Store | 3–4 hrs | Must do |
| Phase 2: Tracking & Registry | 3–4 hrs | Must do |
| Phase 3: Serving Integration | 1–2 hrs | Should do |
| **Total** | **~8–10 hrs** | |

---

## Takeaways

After completing this tutorial, you should be able to speak fluently about:

1. **Feature definitions as code** — why storing feature logic in a repo (not notebooks) matters for reproducibility and team collaboration
2. **Point-in-time correctness** — how Feast prevents future data from leaking into training rows, and why this is non-negotiable in production
3. **Experiment tracking** — the discipline of logging every run with full provenance (params, metrics, artifacts, tags)
4. **Model lifecycle management** — versioning, staging, promotion, and archival as a structured workflow rather than ad hoc file management
5. **The feature store ↔ model server integration** — the pattern where features are defined once and used consistently in both training and inference

---

## Major Gaps to Consider (Beyond This Tutorial)

While this tutorial focuses on core feature management and model lifecycle concepts, here are important production considerations that are intentionally out of scope for a local hands-on tutorial:

### Data Pipeline & Ingestion
- **Data versioning**: How to version datasets used for training (e.g., DVC, S3 versioning)
- **Incremental data updates**: Strategy for handling new data arriving over time
- **Data quality checks**: Validation pipelines to catch schema drift, missing values, or anomalies before training
- **Backfill strategy**: For Feast materialization, how to efficiently backfill historical features when feature definitions change

### Model Validation & Safety
- **Pre-production validation**: Performance thresholds, minimum accuracy requirements before promoting to Production
- **A/B testing framework**: How to compare new model versions against current production models
- **Model monitoring**: Tracking prediction distributions, feature drift, and performance degradation over time
- **Rollback procedures**: Automated rollback triggers when model performance degrades

### Operational Concerns
- **Feature serving latency**: SLA requirements and optimization strategies for online feature retrieval
- **Error handling**: Handling missing features, timeouts, or failures in the feature store or model server
- **Scalability**: How these systems scale beyond local development (distributed feature stores, model serving at scale)
- **Security & access control**: Authentication, authorization, and data privacy considerations

### Development Workflow
- **Testing**: Unit tests for feature definitions, integration tests for the training pipeline, validation tests for model outputs
- **CI/CD**: Automated testing and deployment pipelines (intentionally out of scope for local tutorial)
- **Documentation**: Model cards, feature documentation, and runbook procedures
- **Dependency management**: Using `requirements.txt` or `pyproject.toml` for reproducible environments

### Advanced Feature Store Concepts
- **Feature monitoring**: Detecting feature drift, missing features, or distribution shifts
- **Feature discovery**: How teams discover and reuse existing features
- **Feature lineage**: Tracking feature dependencies and data sources

---

## Suggested Next Step

Walk through your MLE partners' actual feature store and model registry setup with them. This tutorial gives you the vocabulary and mental model; seeing their real configuration will fill in the gaps around scale, CI/CD, orchestration, and monitoring that are hard to simulate locally.
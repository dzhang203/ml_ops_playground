# Phase 1B: Feast Feature Store Walkthrough

This guide walks you through setting up Feast and building a feature-store-backed model. You'll switch between CLI commands, Python files, and a Jupyter notebook.

---

## Prerequisites

- Completed Phase 1A (baseline model trained)
- Have your dataset loaded and understood
- Poetry environment activated (`poetry shell`)

---

## Step 1: Initialize Feast (CLI)

Create the Feast feature repository structure:

```bash
# Create features directory if it doesn't exist
mkdir -p features

# Navigate to it
cd features

# Initialize Feast
poetry run feast init fitness_features
```

**What this does:**
- Creates a `feature_store.yaml` configuration file in your current directory
- Sets up the structure for feature definitions
- Configures local SQLite offline store and file-based online store (default for local dev)

**Expected output:** You'll see a new directory structure with `feature_store.yaml` and example files.

---

## Step 2: Configure Feature Store (File Edit)

Examine the generated `feature_store.yaml`. The defaults should work for local development:
- **Offline store:** SQLite (for training data)
- **Online store:** SQLite (for low-latency serving)
- **Registry:** Local file-based registry

You can keep the defaults or customize if needed. For this tutorial, defaults are fine.

---

## Step 3: Define Features (Python)

**Switch to:** `feature_repo/features.py`

This is where you define:
1. **Entity** - The participant/user (what features are grouped by)
2. **FileSource** - Where Feast reads your raw data from
3. **FeatureView 1** - Rolling heart rate statistics (mean, std over 5min and 15min windows)
4. **FeatureView 2** - Cross-feature interaction (acceleration magnitude × heart rate)

**Implementation task:** Fill in the TODOs in `features.py` to define these components.

**Key concepts:**
- **Entities** represent the "thing" features describe (e.g., participant_id)
- **FeatureViews** are groups of related features with the same data source and entity
- **TTL (time-to-live)** determines how long features are valid in the online store

---

## Step 4: Register Features with Feast (CLI)

Once you've defined your features in `features.py`:

```bash
# Make sure you're in feature_repo/
cd feature_repo

# Apply feature definitions to Feast
feast apply
```

**What this does:**
- Validates your feature definitions
- Registers them in Feast's registry
- Prepares the infrastructure (but doesn't populate data yet)

**Expected output:** Confirmation that entities and feature views were registered.

---

## Step 5: Materialize Features (CLI)

Now populate the offline and online stores with actual feature values:

```bash
# You'll need to determine the timestamp range from your dataset
# Format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
feast materialize <START_TIMESTAMP> <END_TIMESTAMP>
```

**Example:**
```bash
feast materialize 2023-01-01 2023-12-31
```

**What this does:**
- Computes features from your data source
- Writes them to the offline store (for training)
- Optionally writes to online store (for serving)

**Expected output:** Progress bar showing materialization, confirmation of feature counts.

**Tip:** You'll determine the actual timestamp range when you explore your dataset in the notebook.

---

## Step 6: Build Training Dataset with Point-in-Time Joins (Notebook)

**Switch to:** `notebooks/02_feast_features.ipynb`

Now you'll use Feast's `get_historical_features()` to create a training dataset with point-in-time correctness.

**Key workflow:**
1. Create an "entity dataframe" with participant_id, event_timestamp, and labels
2. Call `get_historical_features()` to join features at each timestamp
3. This ensures no future data leaks into training rows

**Implementation task:** Work through the notebook cells to implement this.

---

## Step 7: Train Model and Compare (Notebook)

Still in the notebook:

1. Train LightGBM on the Feast-backed features
2. Evaluate using the same metrics as Phase 1A (accuracy, macro F1, confusion matrix)
3. Compare results to your baseline

**Key insight:** The model performance might not dramatically improve — that's OK! The point is experiencing the workflow and understanding how Feast ensures reproducibility and prevents data leakage.

---

## Step 8: Verify Feature Store State (CLI)

After completing the notebook, you can inspect what Feast has stored:

```bash
# List all feature views
feast feature-views list

# Get details on a specific feature view
feast feature-views describe <feature_view_name>

# List entities
feast entities list
```

**What this does:** Helps you verify that everything was registered and materialized correctly.

---

## Workflow Summary

```
CLI: feast init
  ↓
FILE: Edit feature_store.yaml (optional)
  ↓
PYTHON: Define features in features.py
  ↓
CLI: feast apply (register)
  ↓
CLI: feast materialize (populate)
  ↓
NOTEBOOK: get_historical_features() → train → evaluate
  ↓
CLI: feast feature-views list (verify)
```

---

## Learning Objectives ✓

After completing Phase 1B, you should understand:

- ✓ **Feature definitions as code** - Why `features.py` in version control matters
- ✓ **Point-in-time joins** - How `get_historical_features()` prevents data leakage
- ✓ **Materialization** - The difference between offline store (training) and online store (serving)
- ✓ **Entity-centric design** - How features are organized around entities (participants)

---

## Troubleshooting

**Issue:** `feast apply` fails with validation errors
- Check that your FileSource path is correct
- Verify entity names match between FeatureViews and Entity definitions

**Issue:** `feast materialize` fails or returns 0 features
- Check timestamp range matches your data
- Verify your data source file exists and has the expected schema
- Ensure timestamp column is properly formatted

**Issue:** `get_historical_features()` returns empty dataframe
- Verify your entity dataframe has the correct column names
- Check that timestamps in entity dataframe overlap with materialized feature timestamps

---

## Next Steps

Once Phase 1B is complete, you're ready for **Phase 2: MLflow experiment tracking and model registry**.

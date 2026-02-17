# Feast Feature Definitions for Fitness Activity Classification
#
# This file defines the feature store components:
# - Entity: The "thing" features describe (participant/record)
# - FileSource: Where Feast reads raw data from
# - FeatureViews: Groups of related features
#
# Progress: Entity and FileSource defined. FeatureViews TODO.

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64

# ============================================================================
# ENTITY DEFINITION
# ============================================================================
# Define the record entity - features will be grouped by record_id
#
# ✓ DONE: Created Entity for records
record = Entity(
    name="record",
    join_keys=["record_id"],
    description="Each row is one record, which is a related cluster of fitness device features"
)


# ============================================================================
# DATA SOURCE DEFINITION
# ============================================================================
# Define where Feast should read the raw data from
#
# ✓ DONE: Created FileSource pointing to the dataset
fitness_source = FileSource(
    path="../../../data/full_data_cleaned.csv",
    timestamp_field="synthetic_event_timestamp",
    # Optional: created_timestamp_column="..." (if you have one)
)


# ============================================================================
# FEATURE VIEW 1: ROLLING HEART RATE STATISTICS
# ============================================================================
# Rolling window features capture temporal patterns in heart rate
#
# TODO: Create a FeatureView with rolling heart rate statistics:
# - 5-minute rolling mean of heart rate
# - 5-minute rolling std of heart rate
# - 15-minute rolling mean of heart rate
# - 15-minute rolling std of heart rate
#
# Note: These features should already be computed in your data source.
# The FeatureView just tells Feast which columns to use.
#
# Key parameters:
# - name: Descriptive name for this feature group
# - entities: List of entities these features relate to [record]
# - ttl: How long features are valid (e.g., timedelta(days=1))
# - schema: List of Field() objects describing each feature
# - online: Whether to serve features online (True for this tutorial)
# - source: The FileSource to read from (fitness_source)
# - tags: Optional metadata (e.g., {"team": "ml", "version": "v1"})
#
# Example structure:
# heart_rate_stats = FeatureView(
#     name="heart_rate_stats",
#     entities=[record],
#     ttl=timedelta(days=1),
#     schema=[
#         Field(name="hr_5min_mean", dtype=Float32),
#         Field(name="hr_5min_std", dtype=Float32),
#         Field(name="hr_15min_mean", dtype=Float32),
#         Field(name="hr_15min_std", dtype=Float32),
#     ],
#     online=True,
#     source=fitness_source,
#     tags={"version": "v1"}
# )


# ============================================================================
# FEATURE VIEW 2: CROSS-FEATURE INTERACTION
# ============================================================================
# Cross-feature interactions capture relationships between raw features
#
# TODO: Create a FeatureView for the acceleration-heart rate interaction:
# - acceleration_magnitude: sqrt(accel_x² + accel_y² + accel_z²)
# - exertion_intensity: acceleration_magnitude × heart_rate
#
# This should also already be computed in your data source.
#
# Key insight: This feature represents "how hard is the person moving"
# - High acceleration + high HR = intense activity (running)
# - Low acceleration + low HR = resting
# - High acceleration + low HR = might indicate anomaly or specific activity
#
# Example structure:
# exertion_features = FeatureView(
#     name="exertion_features",
#     entities=[record],
#     ttl=timedelta(days=1),
#     schema=[
#         Field(name="acceleration_magnitude", dtype=Float32),
#         Field(name="exertion_intensity", dtype=Float32),
#     ],
#     online=True,
#     source=fitness_source,
#     tags={"version": "v1"}
# )


# ============================================================================
# NOTES AND TIPS
# ============================================================================
#
# Point-in-time correctness:
# - Feast ensures that when you request features for timestamp T, you only
#   get features computed from data available BEFORE timestamp T
# - This prevents data leakage during training
#
# Materialization:
# - "feast apply" registers these definitions
# - "feast materialize" actually computes and stores the feature values
# - Offline store: Used for training (get_historical_features)
# - Online store: Used for serving (get_online_features)
#
# TTL (Time-to-Live):
# - Determines how stale a feature can be in the online store
# - For activity classification, features older than a few minutes are stale
# - Choose a TTL that makes sense for your use case
#
# Schema fields:
# - Field names must match columns in your FileSource
# - dtypes should match your data (Float32, Float64, Int64, String, etc.)
#
# After defining your FeatureViews:
# 1. Run "feast apply" to register them
# 2. Run "feast materialize <START> <END>" to populate the stores
# 3. Use get_historical_features() in your notebook for training
#
# ============================================================================

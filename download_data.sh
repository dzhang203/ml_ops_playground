#!/bin/bash
# Script to download the Apple Watch and Fitbit dataset from Kaggle

set -e

# Ensure data directory exists
mkdir -p data

# Download the dataset using kaggle CLI
# Dataset: aleespinosa/apple-watch-and-fitbit-data
poetry run kaggle datasets download -d aleespinosa/apple-watch-and-fitbit-data -p data --unzip

echo "Dataset downloaded successfully to data/ directory"

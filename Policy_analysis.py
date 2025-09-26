# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 04:17:13 2025

@author: Francisco Benita
"""

# =============================================================================
# SCRIPT FOR POLICY RECOMMENDATION SYNTHESIS
# =============================================================================
# --- Required Libraries ---
# pandas: For data manipulation
#   conda install -c conda-forge pandas
#
# transformers: For the zero-shot model
#   conda install -c conda-forge transformers
#
# torch: The backend for the transformer model
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# tqdm: For a progress bar during the long classification task
#   conda install -c conda-forge tqdm
#
# seaborn & matplotlib: For creating the heatmap visualization
#   conda install -c conda-forge seaborn matplotlib
# =============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from tqdm import tqdm
import os

# Initialize tqdm to work with pandas .apply()
tqdm.pandas()

# =============================================================================
# --- 1. CONFIGURATION & SETUP ---
# =============================================================================
INPUT_FILE_PATH = r"C:\Users\Articles_Jan2020_Dec2025.csv"
OUTPUT_FOLDER_PATH = r"C:/Users/5-Results_2025/"

# The name for the new CSV file with the classification results
OUTPUT_CSV_NAME = "Articles_with_Policy_Tags.csv"
# The name for the final heatmap image
OUTPUT_FIGURE_NAME = "policy_synthesis_heatmap.png"

# --- Define Classification Labels ---
POLICY_THEME_LABELS = ["Public Health Measures", "Economic Support", "Healthcare System Capacity",
                       "Travel and Border Control", "Communication Strategy", "Vaccination Policy"]

PANDEMIC_STAGE_LABELS = ["Early Stage / Containment", "Peak Stage / Mitigation",
                         "Late Stage / Recovery"]

GOVERNANCE_LEVEL_LABELS = ["National / Federal Level", "Regional / State Level",
                           "City / Municipal Level", "Community / Local Level"]

# We'll only accept classifications with a confidence score above this threshold
CONFIDENCE_THRESHOLD = 0.70

# =============================================================================
# --- 2. ANALYSIS ---
# =============================================================================

if __name__ == "__main__":
    print("--- Starting Policy Synthesis Analysis ---")

    # --- Load Data ---
    try:
        df = pd.read_csv(INPUT_FILE_PATH)
        print(f"Successfully loaded data from: {INPUT_FILE_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
        exit()

    # Handle missing recommendations
    df['Policy recommendations'] = df['Policy recommendations'].fillna('')

    # --- Initialize Classifier ---
    print("\nInitializing Zero-Shot Classification model... (This may take a moment)")
    # Using a distilled, smaller model for better performance on CPU
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

    # --- Define Classification Function ---
    def classify_text(text, labels):
        if not text.strip():
            return ""
        results = classifier(text, labels, multi_label=True)
        # Filter labels by confidence score and join them into a string
        filtered_labels = [l for l, s in zip(results['labels'], results['scores']) if s > CONFIDENCE_THRESHOLD]
        return ', '.join(filtered_labels)

    # --- Run Classification for Each Dimension ---
    label_sets = {
        "Policy_Theme": POLICY_THEME_LABELS,
        "Pandemic_Stage": PANDEMIC_STAGE_LABELS,
        "Governance_Level": GOVERNANCE_LEVEL_LABELS
    }

    for col_name, labels in label_sets.items():
        print(f"\nClassifying by {col_name}...")
        df[col_name] = df['Policy recommendations'].progress_apply(classify_text, labels=labels)

    # --- Save Augmented Data ---
    output_csv_path = os.path.join(OUTPUT_FOLDER_PATH, OUTPUT_CSV_NAME)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully saved augmented data with policy tags to: {output_csv_path}")

# --- Create Synthesis Matrix, Heatmap, and Export Data ---
print("\nGenerating synthesis matrix and heatmap...")

# Focus the spatial analysis on the robust pre-vaccine dataset
pre_vax_df = df[df['available_vaccines'] == 0].copy()


pre_vax_df['Policy_Theme'] = pre_vax_df['Policy_Theme'].str.split(', ')
exploded_df = pre_vax_df.explode('Policy_Theme')

# Clean up empty themes
exploded_df.dropna(subset=['Policy_Theme'], inplace=True)
exploded_df = exploded_df[exploded_df['Policy_Theme'] != '']
exploded_df['Policy_Theme'] = exploded_df['Policy_Theme'].str.strip()


# Create the cross-tabulation matrix
synthesis_matrix = pd.crosstab(exploded_df['Policy_Theme'], exploded_df['small_large'])

# --- Export the synthesis_matrix to a CSV file ---
matrix_output_path = os.path.join(OUTPUT_FOLDER_PATH, "policy_synthesis_matrix.csv")
synthesis_matrix.to_csv(matrix_output_path)
print(f"Successfully saved synthesis matrix to: {matrix_output_path}")

# --- Create and export a table with example policies ---
# For each theme, find the first non-empty policy recommendation
example_policies = exploded_df.groupby('Policy_Theme')['Policy recommendations'].first().reset_index()
example_output_path = os.path.join(OUTPUT_FOLDER_PATH, "policy_examples_by_theme.csv")
example_policies.to_csv(example_output_path, index=False)
print(f"Successfully saved example policies to: {example_output_path}")

# Visualize as a heatmap 
plt.figure(figsize=(12, 8))
sns.heatmap(synthesis_matrix, annot=True, fmt="g", cmap="Blues", linewidths=.5)

plt.title("Policy Theme Synthesis by Geographical Scale (Pre-Vaccine Era)", fontsize=16)
plt.ylabel("Policy Theme", fontsize=12)
plt.xlabel("Geographical Scale", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

output_fig_path = os.path.join(OUTPUT_FOLDER_PATH, OUTPUT_FIGURE_NAME)
plt.savefig(output_fig_path, bbox_inches='tight')
plt.close() # Added to ensure the figure closes properly

print(f"Successfully saved synthesis heatmap to: {output_fig_path}")

print("\n--- Analysis Complete ---")


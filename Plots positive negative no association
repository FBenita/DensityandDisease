# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:18:28 2025

@author: Francisco Benita
"""

# =============================================================================
# SCRIPT FOR DESCRIPTIVE FREQUENCY PLOTS
# =============================================================================
# --- Required Libraries ---
# conda install -c conda-forge pandas seaborn matplotlib
# or: pip install pandas seaborn matplotlib
# =============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =============================================================================
# --- 1. CONFIGURATION & SETUP ---
# =============================================================================
INPUT_FILE_PATH = r"C:\Users\Articles_Jan2020_Dec2025.csv"
OUTPUT_FOLDER_PATH = r"C:/Users/5-Results_2025/"

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

# =============================================================================
# --- 2. DATA LOADING AND PREPARATION ---
# =============================================================================
try:
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"Successfully loaded data from: {INPUT_FILE_PATH}")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
    exit()

# Filter out rows where 'pop_density' is not specified
df_filtered = df.dropna(subset=['pop_density'])
df_filtered = df_filtered[df_filtered['pop_density'] != 'NA']


# =============================================================================
# --- 3. PLOT 1: BY GEOGRAPHICAL SCALE (SGA vs LGA) ---
# =============================================================================
print("Generating plot by geographical scale (SGA/LGA)...")

plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create the count plot
ax1 = sns.countplot(
    data=df_filtered,
    y='small_large',
    hue='pop_density',
    order=df_filtered['small_large'].value_counts().index, # Order by frequency
    palette={'Positive': 'firebrick', 'Negative': 'seagreen', 'No association': 'gold'}
)

# Customize plot
plt.title('Frequency of Findings by Geographical Scale', fontsize=16, weight='bold')
plt.ylabel('Geographical Scale', fontsize=14)
plt.xlabel('Number of Articles', fontsize=14)
plt.legend(title='Association Found')
plt.tight_layout()

# Save the figure
output_path_sga = os.path.join(OUTPUT_FOLDER_PATH, "findings_by_geographical_scale.png")
plt.savefig(output_path_sga)
plt.close()
print(f"Figure saved to: {output_path_sga}")


# =============================================================================
# --- 4. PLOT 2: BY WORLD REGION (PROPORTIONAL) ---
# =============================================================================
print("\nGenerating plot by world region...")

# Calculate proportions for the plot
region_props = df_filtered.groupby('Region')['pop_density'].value_counts(normalize=True).mul(100).rename('proportion').reset_index()

plt.figure(figsize=(12, 8))

# Create a bar plot
ax2 = sns.barplot(
    data=region_props,
    y='Region',
    x='proportion',
    hue='pop_density',
    palette={'Positive': 'firebrick', 'Negative': 'seagreen', 'No association': 'gold'}
)

# Customize plot
plt.title('Proportional Findings by World Region', fontsize=16, weight='bold')
plt.ylabel('World Region', fontsize=14)
plt.xlabel('Proportion of Articles (%)', fontsize=14)
plt.legend(title='Association Found', loc='lower right')
plt.tight_layout()

# Save the figure
output_path_region = os.path.join(OUTPUT_FOLDER_PATH, "findings_by_world_region.png")
plt.savefig(output_path_region)
plt.close()
print(f"Figure saved to: {output_path_region}")

print("\n--- Analysis Complete ---")


# =============================================================================
# --- 5.  EXPORT DATA TO CSV ---
# =============================================================================
print("\nExporting plot data to CSV files...")

# --- Data for Plot 1 (Geographical Scale) ---
# We calculate the counts directly using crosstab for a clean table
scale_counts_df = pd.crosstab(df_filtered['small_large'], df_filtered['pop_density'])
output_csv_scale = os.path.join(OUTPUT_FOLDER_PATH, "data_by_geographical_scale.csv")
scale_counts_df.to_csv(output_csv_scale)
print(f"Data for geographical scale plot saved to: {output_csv_scale}")

# --- Data for Plot 2 (World Region) ---
region_counts_df = pd.crosstab(df_filtered['Region'], df_filtered['pop_density'])
output_csv_region = os.path.join(OUTPUT_FOLDER_PATH, "data_by_world_region.csv")
region_counts_df.to_csv(output_csv_region)
print(f"Data for world region plot saved to: {output_csv_region}")



print("\n--- Analysis Complete ---")

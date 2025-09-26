# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:44:21 2025

@author: Francisco Benita
"""
# =============================================================================
# SCRIPT TO GENERATE WORLD MAP (FINAL, CORRECTED OFFLINE VERSION)
# =============================================================================
# This script uses a downloaded shapefile to create the world map,
# avoiding issues with online packages.

# --- Required Libraries ---
# In your 'covid' conda environment, ensure these are installed:
# conda install -c conda-forge pandas geopandas matplotlib
# =============================================================================

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

# =============================================================================
# --- 1. CONFIGURATION & SETUP ---
# =============================================================================
INPUT_FILE_PATH = r"C:\Users\L03565094\Dropbox\Long+Francisco\5-Data_2025\Articles_Jan2020_Dec2025.csv"
OUTPUT_FOLDER_PATH = r"C:/Users/L03565094/Dropbox/Long+Francisco/5-Results_2025/"

# --- NEW: Set the path to the shapefile you downloaded and unzipped ---
# This path must point to the .shp file specifically.
SHAPEFILE_PATH = r"C:\Users\L03565094\Dropbox\Long+Francisco\5-Data_2025\map_data\ne_110m_admin_0_countries.shp"

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

# =============================================================================
# --- 2. DATA PREPARATION (UPDATED) ---
# =============================================================================
print("--- Generating World Map of Study Frequency ---")

try:
    df = pd.read_csv(INPUT_FILE_PATH)
    print("Successfully loaded data.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
    exit()

# --- NEW LOGIC TO PROCESS THE 'Countries' COLUMN ---
# 1. Use the 'Countries' column as our source.
# 2. Drop rows where the 'Countries' column is empty (NaN).
df.dropna(subset=['Countries'], inplace=True)

# 3. Split the strings in the 'Countries' column by comma.
df['Countries'] = df['Countries'].str.split(', ')

# 4. "Explode" the DataFrame. This creates a new row for each country in a list.
df_exploded = df.explode('Countries')

# 5. Clean up any extra whitespace from country names.
df_exploded['Countries'] = df_exploded['Countries'].str.strip()

# 6. Count the frequency of each country.
country_counts = df_exploded['Countries'].value_counts().reset_index()
country_counts.columns = ['country', 'study_count']
# --- END NEW LOGIC ---

print("\nCountry Counts (Top 10):")
print(country_counts.head(10))

# Standardize country names to match the map data
name_mapping = {
    'United States': 'United States of America',
    'South Korea': 'Republic of Korea',
    'England': 'United Kingdom',
    'Czechia': 'Czechia',
    "Cote d'Ivoire": "Côte d'Ivoire"
}
country_counts['country'] = country_counts['country'].replace(name_mapping)


# =============================================================================
# --- 3. MAP GENERATION ---
# =============================================================================
print("\nLoading world map data from local shapefile...")

try:
    world = geopandas.read_file(SHAPEFILE_PATH)
except Exception as e:
    print(f"ERROR: Could not read the shapefile. Please check the path is correct.")
    exit()

# Merge your study counts with the world map data
merged_map = world.merge(country_counts, left_on='NAME', right_on='country', how='left')
merged_map['study_count'] = merged_map['study_count'].fillna(0)

# --- Plotting the Choropleth Map ---
#print("Generating map visualization...")

#fig, ax = plt.subplots(1, 1, figsize=(20, 12))

#merged_map.plot(
#    column='study_count',
#    cmap='viridis',
#    linewidth=0.8,
#    ax=ax,
#    edgecolor='0.8',
#    legend=True,
#    legend_kwds={
#        'label': "Number of Studies",
#        'orientation': "horizontal",
#        'shrink': 0.6
#    }
#)

# --- Plotting the Choropleth Map ---
print("Generating map visualization...")

fig, ax = plt.subplots(1, 1, figsize=(20, 12))

merged_map.plot(
    column='study_count',
    cmap='plasma', # <-- CHANGE THIS VALUE
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=True,
    legend_kwds={
        'label': "Number of Studies",
        'orientation': "horizontal",
        'shrink': 0.6
    }
)

ax.set_title('Geographical Distribution of Reviewed Studies', fontdict={'fontsize': '24', 'fontweight': '3'})
ax.axis('off')

output_map_path = os.path.join(OUTPUT_FOLDER_PATH, "study_distribution_world_map.png")
plt.savefig(output_map_path, dpi=300)
print(f"Map saved to: {output_map_path}")

print("\n--- Analysis Complete ---")
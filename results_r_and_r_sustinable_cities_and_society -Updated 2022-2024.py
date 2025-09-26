# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:25:07 2025

@author: Francisco Benita
"""

# =============================================================================
# SCRIPT TO GENERATE POPULATION DENSITY DEFINITIONS TABLE
# =============================================================================
# This script replicates the analysis from the original R code to create a
# summary table of how population density is defined in the literature,
# broken down by geographical scale (SGA/LGA).

# --- Required Libraries ---
# conda install -c conda-forge pandas
# or: pip install pandas
# =============================================================================

import pandas as pd
import os

# =============================================================================
# --- 1. CONFIGURATION & SETUP ---
# =============================================================================
INPUT_FILE_PATH = r"C:\Users\L03565094\Dropbox\Long+Francisco\5-Data_2025\Articles_Jan2020_Dec2025.csv"
OUTPUT_FOLDER_PATH = r"C:/Users/L03565094/Dropbox/Long+Francisco/5-Results_2025/"

# The name for the new CSV file with the summary table
OUTPUT_CSV_NAME = "Definitions_density_updated.csv"

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

# =============================================================================
# --- 2. ANALYSIS ---
# =============================================================================
print("--- Generating Population Density Definitions Table ---")

try:
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"Successfully loaded data from: {INPUT_FILE_PATH}")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
    exit()

# Filter the data similarly to the R script
# Assuming 'Included' column exists and you want to filter. If not, comment this line out.
# df = df[df['Included'] == 0] 
df_filtered = df.dropna(subset=['Def_pop_dens', 'small_large'])
df_filtered = df_filtered[df_filtered['small_large'] != 'Others']

# Create the cross-tabulation matrix (equivalent to R's xtabs)
definitions_table = pd.crosstab(df_filtered['Def_pop_dens'], df_filtered['small_large'])

# Order the table by the count in 'Large Geographical Areas', descending
if 'Large Geographical Areas' in definitions_table.columns:
    definitions_table.sort_values(by='Large Geographical Areas', ascending=False, inplace=True)

print("\nRaw Counts Table:")
print(definitions_table)

# --- Format the table with counts and percentages ---
formatted_table = pd.DataFrame(index=definitions_table.index)

for col in definitions_table.columns:
    total_count = definitions_table[col].sum()
    # Create the formatted string: "Count (Percentage%)"
    formatted_table[col] = definitions_table.apply(
        lambda row: f"{row[col]} ({ (row[col]/total_count*100):.2f}%)",
        axis=1
    )

# --- Add a 'Total' row ---
total_row_raw = definitions_table.sum()
total_row_formatted = pd.Series({
    col: f"{total_row_raw[col]} (100.00%)" for col in total_row_raw.index
}, name='Total')

# Use pd.concat to append the total row
formatted_table = pd.concat([formatted_table, pd.DataFrame(total_row_formatted).T])


print("\nFormatted Table with Percentages:")
print(formatted_table)

# --- Save the final table to a CSV file ---
output_path = os.path.join(OUTPUT_FOLDER_PATH, OUTPUT_CSV_NAME)
formatted_table.to_csv(output_path, index=True) # index=True to save the definition names
print(f"\nSuccessfully saved updated definitions table to: {output_path}")

print("\n--- Analysis Complete ---")




# =============================================================================
# SCRIPT TO GENERATE METHODOLOGY TABLE (GEOGRAPHICAL UNIT TABULATION)
# =============================================================================
# This script creates a summary table of the geographical units of analysis
# used in the literature, matching the format of the user's Table 1.

# --- Required Libraries ---
# conda install -c conda-forge pandas
# =============================================================================

import pandas as pd
import os

# =============================================================================
# --- 1. CONFIGURATION & SETUP ---
# =============================================================================
INPUT_FILE_PATH = r"C:\Users\L03565094\Dropbox\Long+Francisco\5-Data_2025\Articles_Jan2020_Dec2025.csv"
OUTPUT_FOLDER_PATH = r"C:/Users/L03565094/Dropbox/Long+Francisco/5-Results_2025/"

# The name for the new CSV file with the summary table
OUTPUT_CSV_NAME = "Geographical_Unit_Tabulation.csv"

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

# =============================================================================
# --- 2. DATA LOADING & CLEANING ---
# =============================================================================
print("--- Generating Geographical Unit Tabulation Table ---")

try:
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"Successfully loaded data from: {INPUT_FILE_PATH}")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
    exit()

# --- Standardize the 'Unit_of_analysis' column ---
unit_normalization_map = {
    'countries': 'Countries', 'countires': 'Countries', 'nations': 'Countries',
    'counties': 'Counties', 'dristricts': 'Districts', 'provices': 'Provinces',
    'province': 'Provinces', 'states': 'States',
    'middle layer super output area': 'Middle Layer Super Output Areas',
    'local authorities': 'Local Authorities', 'residential units': 'Residential Units',
    'urban area': 'Urban Areas'
}
# Create a new cleaned column
df['Unit_of_analysis_Clean'] = df['Unit_of_analysis'].str.lower().str.strip()
df['Unit_of_analysis_Clean'] = df['Unit_of_analysis_Clean'].replace(unit_normalization_map)
df['Unit_of_analysis_Clean'] = df['Unit_of_analysis_Clean'].str.capitalize()

# --- Add a more descriptive category name for 'Others' ---
df['small_large'] = df['small_large'].replace({'Others': 'Non-Geographical Units'})
print("Cleaned 'Unit_of_analysis' column and renamed 'Others' category.")

# --- Create a helper column for counting policy recommendations ---
df['has_policy_recommendation'] = df['Policy recommendations'].notna() & (df['Policy recommendations'] != '')

# =============================================================================
# --- 3. ANALYSIS & TABLE GENERATION ---
# =============================================================================

def create_panel(category_name, df_full):
    """Processes data for a single panel of the final table."""
    df_cat = df_full[df_full['small_large'] == category_name].copy()
    
    # Group by the new 'Unit_of_analysis_Clean' column
    summary = df_cat.groupby('Unit_of_analysis_Clean').agg(
        Total=('ID', 'count'),
        TWPR=('has_policy_recommendation', 'sum')
    ).astype(int)
    
    # Sort the summary table by total count, descending
    summary.sort_values(by='Total', ascending=False, inplace=True)
    
    # Calculate and format the TWPR column as "Count (Percentage%)"
    summary['TWPR_Percent'] = (summary['TWPR'] / summary['Total'] * 100).round(1)
    summary['TWPR*'] = summary.apply(
        lambda row: f"{row['TWPR']} ({row['TWPR_Percent']}%)" if row['Total'] > 0 else "0 (0.0%)",
        axis=1
    )
    
    # Add a total row
    total_row = pd.DataFrame({
        'Total': summary['Total'].sum(),
        'TWPR': summary['TWPR'].sum()
    }, index=['Total'])
    if total_row['Total'].iloc[0] > 0:
        total_row['TWPR_Percent'] = (total_row['TWPR'] / total_row['Total'] * 100).round(1)
        total_row['TWPR*'] = total_row.apply(lambda row: f"{row['TWPR']} ({row['TWPR_Percent']}%)", axis=1)
    else:
        total_row['TWPR*'] = "0 (0.0%)"
    
    # Combine the sorted summary with the total row
    final_panel = pd.concat([summary, total_row])
    
    # Select and rename columns for the final table
    final_panel = final_panel[['Total', 'TWPR*']]
    final_panel.rename(columns={'Total': f'{category_name}_Total', 'TWPR*': f'{category_name}_TWPR*'}, inplace=True)
    final_panel.index.name = category_name
    
    return final_panel

# --- Generate each panel ---
sga_panel = create_panel('Small Geographical Areas', df)
lga_panel = create_panel('Large Geographical Areas', df)
others_panel = create_panel('Non-Geographical Units', df) # Use the new descriptive name

# --- Combine all panels into one final table ---
final_table = pd.concat([sga_panel, lga_panel, others_panel], axis=1)

print("\nGenerated Final Table:")
print(final_table)

# --- Save the final table to a CSV file ---
output_path = os.path.join(OUTPUT_FOLDER_PATH, OUTPUT_CSV_NAME)
final_table.to_csv(output_path, index=True)
print(f"\nSuccessfully saved the final cleaned table to: {output_path}")

print("\n--- Analysis Complete ---")

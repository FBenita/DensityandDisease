# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 02:37:28 2025

@author: Francisco Benita
"""
# =============================================================================
# SCRIPT FOR TEMPORAL KEYWORD NETWORK ANALYSIS
# =============================================================================
# This script loads review data, cleans keywords, separates the data into
# pre-vaccine and post-vaccine periods, and generates keyword co-occurrence
# network visualizations for each period.

# --- Required Libraries ---
# conda install -c conda-forge pandas networkx matplotlib
# or: pip install pandas networkx matplotlib
# =============================================================================

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from itertools import combinations
from collections import Counter
import os
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.stats import gaussian_kde
# =============================================================================
# --- 1. CONFIGURATION & SETUP ---
# =============================================================================
# IMPORTANT: Double-check that these file paths are correct for your system.
INPUT_FILE_PATH = r"C:\Users\L03565094\Dropbox\Long+Francisco\5-Data_2025\Articles_Jan2020_Dec2025.csv"
OUTPUT_FOLDER_PATH = r"C:/Users/L03565094/Dropbox/Long+Francisco/5-Results_2025/"

TOP_N_KEYWORDS = 40

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)
    print(f"Created output folder: {OUTPUT_FOLDER_PATH}")

# =============================================================================
# --- 2. DATA PROCESSING & NETWORK FUNCTIONS ---
# =============================================================================

# Define your cleaning rules
normalization_map = {
    # (Your COVID-19 and Population Density terms remain the same)
    "coronavirus": "covid-19", "sars-cov-2": "covid-19",
    "population density": "population-density",
  
    
    # Geographic terms (standardize but don't remove from network)
    "united states": "usa", "u.s.": "usa", "united kingdom": "uk",

    # ---  Standardize Specific Methodological Terms ---
    "gwr model": "gwr",
    "geographically weighted regression": "gwr",
    "geographically weighted regression (gwr)": "gwr",
    "artificial intelligence": "ai",
    "machine learning": "ai",
    "geographic information systems": "gis",
    "clustering": "clustering",
    "clusterings": "clustering",
    "built": "built environment",
   
    # ---  Standardize city terms ---
    "cities": "city",
    "city": "city"
}

stop_words = {
    # --- Core topics (too obvious to plot as nodes) ---
    "covid-19", "19", "covid", "population-density", "density","pandemics",
    "betacoronavirus", "pneumonia, viral",
    
    # --- NEW: Country & Region names (covered by the map) ---
    "usa", "india", "china", "canada", "hong kong", "africa",
    "uk", "turkey", "italy", "iran", "spain", "brazil", "germany",
    "europe", "asia", "north america", "african", "counties", "city",
    
    # --- Redundant / overly broad terms ---
    "pandemics", "disease transmission", "disease spread",
    "coronavirus disease 2019", "severe acute respiratory syndrome",
    "pandemic", "epidemic", "communicable disease control",
    "severe acute respiratory syndrome coronavirus 2",
    "coronaviruses", "incidence", "coronavirus infections", "infection",
    
    # --- Generic methodological terms (keep these out) ---
    "article", "human", "humans", "methods", "data", "regression","risk", "model",
    "regression analysis", "cluster analysis", "spatial analysis",
    "risk factor", "risk assessment", "population statistics", "risk factors",
    "principal component analysis", "results", "study", "impact", "conclusions"
}

#def process_keywords(df):
#    """Cleans, normalizes, and processes the 'Keywords' column."""
#    df = df.dropna(subset=['Keywords'])
#    keyword_lists = []
#    for _, row in df.iterrows():
#        raw_keywords = [kw.strip().lower() for kw in str(row['Keywords']).split(';')]
#        normalized_keywords = [normalization_map.get(kw, kw) for kw in raw_keywords]
#        cleaned_keywords = {kw for kw in normalized_keywords if kw and kw not in stop_words}
#        if cleaned_keywords:
#            keyword_lists.append(list(cleaned_keywords))
#    return keyword_lists

def process_keywords(df):
    """
    HYBRID APPROACH: Cleans and processes both the 'Keywords' column and
    extracts the top N keywords from the 'Abstract' column using TF-IDF.
    """
    N_TOP_TERMS_FROM_ABSTRACT = 5
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    df_nonan = df.dropna(subset=['Abstract']).copy()
    
    if df_nonan.empty:
        return []
        
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_nonan['Abstract'])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    keyword_lists = []
    for index, row in df.iterrows():
        author_keywords = set()
        if pd.notna(row['Keywords']):
            raw_keywords = [kw.strip().lower() for kw in str(row['Keywords']).split(';')]
            author_keywords = {normalization_map.get(kw, kw) for kw in raw_keywords}

        machine_keywords = set()
        if pd.notna(row['Abstract']) and index in df_nonan.index:
            doc_index = df_nonan.index.get_loc(index)
            feature_index = tfidf_matrix[doc_index,:].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[doc_index, x] for x in feature_index])
            top_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:N_TOP_TERMS_FROM_ABSTRACT]
            machine_keywords = {feature_names[i] for i, _ in top_terms}

        combined_keywords = author_keywords.union(machine_keywords)
        final_keywords = [kw for kw in combined_keywords if kw and kw not in stop_words]
        
        if final_keywords:
            keyword_lists.append(final_keywords)
            
    return keyword_lists

def build_network(keyword_lists, top_n):
    """Builds a co-occurrence network from a list of keyword lists."""
    all_keywords = [kw for sublist in keyword_lists for kw in sublist]
    keyword_counts = Counter(all_keywords)
    top_keywords = {kw for kw, _ in keyword_counts.most_common(top_n)}
    filtered_lists = [[kw for kw in sublist if kw in top_keywords] for sublist in keyword_lists]
    
    G = nx.Graph()
    co_occurrences = Counter()
    for sublist in filtered_lists:
        for pair in combinations(sorted(sublist), 2):
            co_occurrences[pair] += 1
            
    for (kw1, kw2), weight in co_occurrences.items():
        if weight > 0:
            G.add_edge(kw1, kw2, weight=weight)
            
    for node in G.nodes():
        G.nodes[node]['frequency'] = keyword_counts.get(node, 0)
    return G

def visualize_network(G, title, file_path):
    """
    Creates, styles, finds communities, and saves the network visualization
    with an improved color palette and more legible labels.
    """
    if not G.nodes():
        print(f"Skipping visualization for '{title}' as no network was generated.")
        return
        
    print(f"Generating visualization for: {title}...")
    
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    partition = community_louvain.best_partition(G, weight='weight')
    
    # --- MODIFIED: Use a lighter, more distinct color palette ---
    # Palettes like 'pastel1', 'Set3', or 'tab20' are better for this.
    cmap = plt.get_cmap('Set3', max(partition.values()) + 1)
    
    node_colors = [cmap(partition[node]) for node in G.nodes()]

    node_sizes = [d['frequency'] * 100 for _, d in G.nodes(data=True)]
    edge_widths = [d['weight'] * 0.5 for _, _, d in G.edges(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.6)
    
    # --- NEW: Draw labels with a white outline for readability ---
    labels = nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    for _, t in labels.items():
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    # --- END NEW SECTION ---
    
    plt.title(title, fontsize=30)
    plt.axis('off')
    plt.savefig(file_path, format="PNG", bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {file_path}")

def visualize_density_network(G, title, file_path):
    """
    Creates a VOSviewer-style density map visualization of the network.
    """
    if not G.nodes():
        print(f"Skipping density map for '{title}' as no network was generated.")
        return

    print(f"Generating density map for: {title}...")
    
    # Use the same layout to ensure consistency with the node-link plot
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    # Extract node positions and frequencies (weights for the density)
    x = [p[0] for p in pos.values()]
    y = [p[1] for p in pos.values()]
    weights = [d['frequency'] for _, d in G.nodes(data=True)]

    # Create a Kernel Density Estimate (KDE)
    xy = np.vstack([x,y])
    kde = gaussian_kde(xy, weights=weights)

    # Create a grid to evaluate the density on
    x_grid, y_grid = np.mgrid[min(x)-0.1:max(x)+0.1:100j, min(y)-0.1:max(y)+0.1:100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = np.reshape(kde(positions).T, x_grid.shape)

    # Plot the density map
    plt.figure(figsize=(20, 20))
    plt.contourf(x_grid, y_grid, z, levels=20, cmap='Blues')
    plt.contour(x_grid, y_grid, z, levels=20, linewidths=0.5, colors='black')
    
    plt.title(title, fontsize=30)
    plt.axis('off')
    plt.savefig(file_path, format="PNG", bbox_inches='tight')
    plt.close()
    print(f"Density map saved to: {file_path}")

    
def visualize_vosviewer_style(G, title, file_path):
    """
    Creates a VOSviewer-style density visualization with large, transparent nodes
    and more visible edges.
    """
    if not G.nodes():
        print(f"Skipping VOSviewer-style plot for '{title}' as no network was generated.")
        return

    print(f"Generating VOSviewer-style plot for: {title}...")
    
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    partition = community_louvain.best_partition(G, weight='weight')
    cmap = plt.get_cmap('Set3', max(partition.values()) + 1)
    node_colors = [cmap(partition[node]) for node in G.nodes()]

    # Node styling remains the same
    node_sizes = [d['frequency'] * 800 for _, d in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.4)
    
    # --- MODIFIED: Make edges darker and more opaque ---
    nx.draw_networkx_edges(G, pos, width=0.3, edge_color='black', alpha=0.2) # Changed color and alpha
    
    # Label styling remains the same
    labels = nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', font_color='black')
    for _, t in labels.items():
        t.set_alpha(0.7)
    
    plt.title(title, fontsize=30)
    plt.axis('off')
    plt.savefig(file_path, format="PNG", bbox_inches='tight')
    plt.close()
    print(f"VOSviewer-style plot saved to: {file_path}")
    
    
# =============================================================================
# --- 3. MAIN EXECUTION & NEW DATA EXPORT ---
# =============================================================================

if __name__ == "__main__":
    print("--- Starting Temporal Keyword Network Analysis ---")
    
    try:
        df = pd.read_csv(INPUT_FILE_PATH)
        print(f"Successfully loaded data from: {INPUT_FILE_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
        exit()
        
    pre_vax_df = df[df['available_vaccines'] == 0].copy()
    post_vax_df = df[df['available_vaccines'] == 1].copy()
    print(f"Split data: {len(pre_vax_df)} pre-vaccine papers, {len(post_vax_df)} post-vaccine papers.")

    # --- NEW FUNCTION TO EXPORT NETWORK DATA ---
    def export_network_data(G, partition, file_path):
        """Exports node and edge data from the network to a CSV file."""
        if not G.nodes():
            print(f"Skipping data export for {file_path} as the network is empty.")
            return

        # Create a dataframe for the nodes
        node_data = []
        for node, data in G.nodes(data=True):
            node_data.append({
                'Keyword': node,
                'Frequency': data.get('frequency', 0),
                'Cluster': partition.get(node, -1) + 1 # Add 1 to make clusters start from 1
            })
        nodes_df = pd.DataFrame(node_data).sort_values(by=['Cluster', 'Frequency'], ascending=[True, False])

        # Save to CSV
        nodes_df.to_csv(file_path, index=False)
        print(f"Network data exported to: {file_path}")

    # --- Process and Visualize Pre-Vaccine Data ---
    print("\n--- Processing Pre-Vaccine Period ---")
    pre_vax_keywords = process_keywords(pre_vax_df)
    pre_vax_network = build_network(pre_vax_keywords, top_n=TOP_N_KEYWORDS)
    
    # Generate visualizations (as before)
    visualize_network(pre_vax_network, 
                      f"Keyword Network (Pre-Vaccine Era, Top {TOP_N_KEYWORDS})", 
                      os.path.join(OUTPUT_FOLDER_PATH, "keyword_network_pre_vaccine_NODES.png"))
    visualize_density_network(pre_vax_network,
                              f"Keyword Density Map (Pre-Vaccine Era)",
                              os.path.join(OUTPUT_FOLDER_PATH, "keyword_network_pre_vaccine_DENSITY.png"))
    visualize_vosviewer_style(pre_vax_network,
                              f"VOSviewer Style Density (Pre-Vaccine Era)",
                              os.path.join(OUTPUT_FOLDER_PATH, "keyword_network_pre_vaccine_VOS.png"))
    
    # --- NEW: Export pre-vaccine network data ---
    pre_vax_partition = community_louvain.best_partition(pre_vax_network, weight='weight')
    export_network_data(pre_vax_network, pre_vax_partition, 
                        os.path.join(OUTPUT_FOLDER_PATH, "network_data_pre_vaccine.csv"))

    # --- Process and Visualize Post-Vaccine Data ---
    print("\n--- Processing Post-Vaccine Period ---")
    post_vax_keywords = process_keywords(post_vax_df)
    post_vax_network = build_network(post_vax_keywords, top_n=TOP_N_KEYWORDS)
    
    # Generate visualizations (as before)
    visualize_network(post_vax_network, 
                      f"Keyword Network (Post-Vaccine Era, Top {TOP_N_KEYWORDS})", 
                      os.path.join(OUTPUT_FOLDER_PATH, "keyword_network_post_vaccine_NODES.png"))
    visualize_density_network(post_vax_network,
                              f"Keyword Density Map (Post-Vaccine Era)",
                              os.path.join(OUTPUT_FOLDER_PATH, "keyword_network_post_vaccine_DENSITY.png"))
    visualize_vosviewer_style(post_vax_network,
                              f"VOSviewer Style Density (Post-Vaccine Era)",
                              os.path.join(OUTPUT_FOLDER_PATH, "keyword_network_post_vaccine_VOS.png"))

    # --- NEW: Export post-vaccine network data ---
    post_vax_partition = community_louvain.best_partition(post_vax_network, weight='weight')
    export_network_data(post_vax_network, post_vax_partition, 
                        os.path.join(OUTPUT_FOLDER_PATH, "network_data_post_vaccine.csv"))
    
    print("\n--- Analysis Complete ---")
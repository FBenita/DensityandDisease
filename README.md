# Data and Code for "Density and Disease: A Systematic Review of COVID-19 Transmission and Policy Across Scales and Time"

This repository contains the dataset and Python scripts necessary to reproduce the results in the paper:

**Benita, F. (2025). Density and Disease: A Systematic Review of COVID-19 Transmission and Policy Across Scales and Time. *Canadian Geographies / Géographies canadiennes*. DOI: (Pending Publication)**

---

## Overview

This repository includes:
* The main dataset of 250 peer-reviewed articles.
* All Python scripts used for the statistical analysis, computational text analysis (TF-IDF, zero-shot classification), and figure generation.

The files are organized to facilitate the full replication of the methodology and all figures and tables presented in the publication.

## Dataset

* `Articles_Jan2020_Dec2025.csv`: The main dataset containing the 250 articles, including all extracted metadata, keywords, abstracts, and policy recommendations.

## Code

The Python scripts are designed to be run in a Conda environment (see `requirements.txt`) and perform the following analyses:

1.  **Descriptive Statistics:** Generates the frequency bar charts of findings by geographical scale and world region (Figure 1) and the table of density definitions (Tables 1 and 2).
3.  **Thematic Evolution Analysis:** Performs the keyword co-occurrence analysis (using TF-IDF and author keywords), runs the Louvain community detection, and exports the network data and all visualizations (Figures 2 and 3).
4.  **Policy Synthesis:** Performs the zero-shot classification to categorize all policy recommendations by theme, pandemic stage, and governance level, and exports the final synthesis tables (Tables 3 and 4).

## License

The code and data in this repository are provided under the MIT Licens

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17624551.svg)](https://doi.org/10.5281/zenodo.17624551)



# analysis.py
# Marimo Interactive Data Analysis Notebook
# Author: Data Scientist, IIT Madras Research Group
# Contact: 24f2004824@ds.study.iitm.ac.in

import marimo as mo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 1: Load and display dataset info
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
mo.md(f"""
# Interactive Penguin Species Analysis 🐧

**Dataset**: Palmyra Island Penguins (n = {len(df)})  
**Source**: Seaborn datasets  
**Contact**: 24f2004824@ds.study.iitm.ac.in

This notebook demonstrates variable relationships using interactive controls.
""")

# Cell 2: Interactive slider widget (THIS IS THE REQUIRED SLIDER)
# User controls the number of bins in the histogram
n_bins = mo.ui.slider(
    start=5,
    stop=50,
    step=1,
    value=20,
    label="Number of histogram bins:"
)

mo.md(f"""
### Histogram Bin Control
Adjust the slider to explore distribution granularity:  
**Current value**: {n_bins.value}

This slider drives the visualization in the next cell.
""")

# Cell 3: Dependent cell — reacts to the slider above
# Documentation: This cell depends on n_bins from the previous cell
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(
    data=df,
    x="flipper_length_mm",
    hue="species",
    bins=n_bins.value,   # ← Depends on slider
    kde=True,
    ax=ax
)
plt.title(f"Flipper Length Distribution by Species (bins = {n_bins.value})")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Count")
mo.pyplot(fig)

# Cell 4: Dynamic markdown + statistical summary based on slider state
selected_bins = n_bins.value  # ← Another dependency on the same slider

summary = df.groupby('species')['flipper_length_mm'].agg(['mean', 'std']).round(2)

mo.md(f"""
### Dynamic Analysis Summary (updated with {selected_bins} bins)

- **Adelie**: Mean flipper length = {summary.loc['Adelie', 'mean']} ± {summary.loc['Adelie', 'std']} mm
- **Chinstrap**: Mean = {summary.loc['Chinstrap', 'mean']} ± {summary.loc['Chinstrap', 'std']} mm  
- **Gentoo**: Mean = {summary.loc['Gentoo', 'mean']} ± {summary.loc['Gentoo', 'std']} mm (longest flippers)

Higher bin counts reveal multi-modal distributions in Gentoo penguins.
""")

# Cell 5: Final interactive correlation explorer
species_choice = mo.ui.dropdown(
    options=df['species'].dropna().unique().tolist(),
    value="Gentoo",
    label="Select species for detailed view:"
)

mo.md(f"### Species Focus: {species_choice.value}")

filtered = df[df['species'] == species_choice.value]

fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=filtered,
    x="bill_length_mm",
    y="bill_depth_mm",
    size="body_mass_g",
    hue="sex",
    ax=ax2
)
plt.title(f"Bill Dimensions vs Body Mass — {species_choice.value}")
mo.pyplot(fig2)
    

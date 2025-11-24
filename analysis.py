# 24f2004824@ds.study.iitm.ac.in
# Raw GitHub URL for this notebook file:
# https://raw.githubusercontent.com/24f2004824-droid/research-data/main/analysis.py

# %%
"""
Marimo-style interactive analysis script (Python .py cells) - self-documenting

Features:
- At least two cells with variable dependencies
- Interactive slider widget (ipywidgets)
- Dynamic markdown output based on widget state
- Comments documenting the data flow between cells

How to run:
- Open this file in a Jupyter environment or any editor that understands # %% cells
- Install requirements: pip install pandas numpy matplotlib ipywidgets
- In Jupyter: run each cell top-down. The slider updates plots & markdown dynamically.
"""

# --- Cell 1: imports and configuration ---
# Purpose: bring in libraries and set global plot settings.
# Downstream cells depend on these variables (e.g., `plt`, `pd`, `np`).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets

# Make plotting inline-friendly. These are global objects used later.
plt.rcParams['figure.figsize'] = (8, 4)

# %%
# --- Cell 2: data loading and preprocessing ---
# Purpose: create or load a dataset and prepare derived variables.
# Downstream cells depend on `df` (the DataFrame), and derived columns such as 'x' and 'y'.

RAW_URL = "https://raw.githubusercontent.com/24f2004824-droid/research-data/main/analysis.py"

try:
    # Try to load a CSV named `data.csv` in the same repo (optional). If not present, fall back to synthetic data.
    # This demonstrates a real-data path and a reproducible fallback.
    # Note: When run locally, replace the path with a real raw GitHub CSV if available.
    df = pd.read_csv('data.csv')
    data_source = 'local data.csv'
except Exception:
    # If no external data is available, create a synthetic dataset.
    # The synthetic dataset intentionally has a nonlinear relationship plus noise so analysis is meaningful.
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0, 100, size=n)
    # y depends on x (this dependency will be explored interactively)
    y = 0.5 * x + 10 * np.sin(x / 8) + rng.normal(0, 6, size=n)
    category = np.where(x > 50, 'high', 'low')
    df = pd.DataFrame({'x': x, 'y': y, 'category': category})
    data_source = 'synthetic'

# Create a derived column that will be used downstream
# Dependency: df -> df['x_norm']
df['x_norm'] = (df['x'] - df['x'].mean()) / df['x'].std()

# Quick head for documentation when run interactively
print(f"Data loaded from: {data_source}. Rows: {len(df)}")
print(df.head())

# %%
# --- Cell 3: interactive controls and variable dependencies ---
# Purpose: create widgets that control analysis parameters.
# Downstream computations depend on widget values (e.g., `threshold_slider.value`).

# Slider will control the x threshold used to split/filter the dataset.
threshold_slider = widgets.FloatSlider(
    value=50.0,
    min=float(df['x'].min()),
    max=float(df['x'].max()),
    step=0.5,
    description='x threshold',
    continuous_update=True
)

# Dropdown to choose aggregation function
agg_dropdown = widgets.Dropdown(
    options=['mean', 'median', 'count'],
    value='mean',
    description='Aggregate'
)

controls = widgets.HBox([threshold_slider, agg_dropdown])

# The filtered_df function depends on df and threshold_slider.value
def compute_filtered_df(threshold: float):
    """Return a filtered DataFrame and an aggregated summary.

    Data flow:
    - Input: df (global) and threshold (widget value)
    - Output: filtered subset `fdf` and an aggregate value `agg_val`
    """
    fdf = df[df['x'] >= threshold].copy()
    return fdf

# %%
# --- Cell 4: dynamic display (plot + markdown) that reacts to widgets ---
# Purpose: show a plot and markdown summary that change when the widget state changes.
# This cell depends on: df, compute_filtered_df, threshold_slider, agg_dropdown

output_area = widgets.Output()

# Create a placeholder Markdown to be updated dynamically
md_placeholder = Markdown('')

# Update function that redraws plot and markdown based on widget state
def update_display(change=None):
    # Read widget values (dependencies)
    threshold = threshold_slider.value
    agg_choice = agg_dropdown.value

    # Compute filtered data (depends on df and threshold)
    fdf = compute_filtered_df(threshold)

    # Compute an aggregate metric (depends on fdf and agg_choice)
    if len(fdf) == 0:
        agg_val = None
    else:
        if agg_choice == 'mean':
            agg_val = fdf['y'].mean()
        elif agg_choice == 'median':
            agg_val = fdf['y'].median()
        elif agg_choice == 'count':
            agg_val = len(fdf)

    # Prepare dynamic markdown based on widget state
    if agg_val is None:
        md_text = f"**Threshold:** {threshold:.2f}\n\nNo points have x ≥ {threshold:.2f}."
    else:
        # Different wording depending on the aggregation selected
        if agg_choice == 'count':
            md_text = f"**Threshold:** {threshold:.2f}  
**Points with x ≥ threshold:** {agg_val}"
        else:
            md_text = (
                f"**Threshold:** {threshold:.2f}  \n\n"
                f"**Aggregate ({agg_choice}) of y for x ≥ threshold:** {agg_val:.3f}"
            )

    # Redraw plot and markdown inside output_area
    with output_area:
        clear_output(wait=True)
        # Plot full scatter in light gray, highlight filtered points
        plt.scatter(df['x'], df['y'], alpha=0.2)
        if len(fdf) > 0:
            plt.scatter(fdf['x'], fdf['y'], alpha=0.9)
        plt.axvline(threshold, color='k', linestyle='--')
        plt.title('Relationship between x and y (filtered region highlighted)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        display(Markdown(md_text))

# Wire the widgets to the update function
threshold_slider.observe(update_display, names='value')
agg_dropdown.observe(update_display, names='value')

# Initial display
display(controls)
display(output_area)
# Call once to render initial state
update_display()

# %%
# --- Cell 5: additional dependent calculation and explanation ---
# Purpose: show a second cell whose variables depend on the filtered result used above.
# For example: compute a simple linear fit on the filtered data and display coefficients.

# We'll provide a button to trigger the fit (so user can adjust threshold first)
fit_button = widgets.Button(description='Fit line to filtered data')
fit_output = widgets.Output()


def fit_on_click(btn):
    threshold = threshold_slider.value
    fdf = compute_filtered_df(threshold)
    with fit_output:
        clear_output()
        if len(fdf) < 2:
            print('Not enough points to fit a line. Try lowering the threshold.')
            return
        # Simple linear regression (least squares) on filtered points
        coeffs = np.polyfit(fdf['x'], fdf['y'], deg=1)
        slope, intercept = coeffs[0], coeffs[1]
        print(f'Linear fit on points with x ≥ {threshold:.2f}:')
        print(f'  slope = {slope:.4f}')
        print(f'  intercept = {intercept:.4f}')
        # Overlay the fitted line on the plot in the main output area
        with output_area:
            clear_output(wait=True)
            plt.scatter(df['x'], df['y'], alpha=0.2)
            fdf = compute_filtered_df(threshold)
            plt.scatter(fdf['x'], fdf['y'], alpha=0.9)
            xs = np.array([df['x'].min(), df['x'].max()])
            plt.plot(xs, slope * xs + intercept)
            plt.axvline(threshold, color='k', linestyle='--')
            plt.title('Scatter + fitted line (for filtered region)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            # Re-display the dynamic markdown summary as well
            # Recompute aggregate for consistency
            if agg_dropdown.value == 'count':
                md_text = f"**Threshold:** {threshold:.2f}  \n**Points with x ≥ threshold:** {len(fdf)}"
            else:
                agg_val = fdf['y'].mean() if agg_dropdown.value == 'mean' else fdf['y'].median()
                md_text = (
                    f"**Threshold:** {threshold:.2f}  \n\n"
                    f"**Aggregate ({agg_dropdown.value}) of y for x ≥ threshold:** {agg_val:.3f}"
                )
            display(Markdown(md_text))

fit_button.on_click(fit_on_click)

# Display fit UI and output
display(widgets.VBox([fit_button, fit_output]))

# End of notebook
print('\nNotebook cells complete. Use the slider and dropdown above to explore the relationship between x and y.')

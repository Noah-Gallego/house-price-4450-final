# CMPS 4450 House Price Project

Data mining final project studying house-price prediction with property specifications, city information, spatial features, image-derived features, and clustering.

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,numpy,pandas&theme=light" alt="Python, NumPy, and pandas" />
</p>

## Research workflow

The project is organized as a procedural Python analysis pipeline. It includes exploratory analysis, baseline models, shuffled cross-validation, error analysis, tuned trees and KNN models, clustering, geocoding, spatial features, filter-bank experiments, and final evaluation.

## Repository contents

- `scripts/` — feature extraction, analysis, model evaluation, visualization, geocoding, and slide-building scripts
- `figures/` — generated plots and analysis figures
- `results/` — saved metrics, comparisons, clustering summaries, and evaluation artifacts
- `slides_assets/` — presentation assets
- `HousePrice_Final_Project.pdf` and `HousePrice_Final_Project.pptx` — final deliverables

## Methods

The analysis uses vanilla Python data-science tooling, including pandas, NumPy, scikit-learn, Matplotlib, SciPy, and PIL. It compares linear regression, decision trees, KNN, image-derived features, spatial features, and clustering without deep learning or pretrained features.

## Evidenced test results

The saved final test metrics contain 3,095 test examples. The best listed test MAE is `$142,380` for the depth-15 tree using specification features; its RMSE is `$217,806` and 32.9% of predictions are within `$50,000`.

These are recorded experiment results, not a claim that the model is suitable for real-world valuation.

## Reproduction

The scripts are numbered to reflect the analysis sequence. Inspect each script's input paths and configuration before running it; the repository does not provide a single dependency manifest or one verified end-to-end command.

## Context and attribution

This repository is a CMPS 4450 Data Mining final project at CSU Bakersfield. The dataset and included third-party material retain their original rights and attribution. No license is declared.

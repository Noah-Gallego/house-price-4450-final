# CMPS 4450 — Data Mining Final Project

## Context
- Course: CMPS 4450 (Data Mining), CSUB, Dr. Chengwei Lei
- Student: Noah Gallego
- Deliverable: pick a dataset, pick a research question, tell the full story end-to-end. Dr. Lei weights storytelling heavily — the deck has to walk the audience from a plain question to a plain answer with no jumps.
- Constraint: nearly all vanilla Python (pandas, numpy, scikit-learn, matplotlib, scipy, PIL). No deep learning, no pretrained features.

## Coding conventions (strict)
- No file-level comments or docstrings of any kind. No `"""..."""` blocks anywhere.
- Inline comments only — short, no periods, only where intent isn't obvious from the code.
- No AI-style narrating comments (`# Fit the model`, `# Load data`, etc.).
- No OOP. Procedural functions only.

## Voice and framing (strict)
- Write in plain direct language. No editorial or marketing voice. Don't write things like "X say no", "the model is cheating", "the answer is...", or any punchy reveal. The user finds this voice actively bad.
- Slide titles, captions, and CLAUDE.md prose state findings flatly. No setups, no reveals, no clever framings. Plain.
- Reference deck for tone, format, and section structure: `/home/noah-gallego/Dropbox/Desktop/KNN-Midterm/KNN_Presentation.pptx`. Lowercase prose, comma-separated clauses, section headers like "how we set it up", "why this design", "the idea", "what we noticed", "takeaway".
- Slide pattern from that deck: short title line, one-line descriptor underneath, then body broken into 2–3 named sections.

## Evaluation metrics policy
All model evaluation uses interpretable absolute-unit metrics: MAE (dollars), RMSE (dollars), SSE, median absolute error (dollars), max absolute error (dollars), and "% of predictions within $50k / $100k / $200k". No R² or normalized variance scores.

Price is right-skewed but bounded at $195k–$2M (Kaggle source clipped the long tail), so we regress on raw price and report errors in dollars. No log-transform.

## Train / val / test split policy
Every model uses a three-way split: **65% train, 15% validation, 20% test**, with `random_state=42`. Roles are strict.
- **Train (~10,058 rows)** — fit parameters.
- **Validation (~2,321 rows)** — every modeling decision (feature inclusion, model class, hyperparameters).
- **Test (~3,095 rows)** — touched once, after all decisions are locked.

## Dataset
House Prices and Images - SoCal — https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal

15,474 Southern California house listings. One exterior photo per house plus tabular fields: `image_id`, `street`, `citi` (city), `n_citi` (label-encoded city), `bed`, `bath`, `sqft`, `price`. No missing values. Price range $195,000 to $2,000,000, mean $703k, median $639k. 415 unique cities; top 10 cities cover ~25% of listings.

## Research question
What can a house's exterior photo tell us about its price on top of plain specs (beds, baths, sqft, city)?

## Feature set
We predict `price` from a combination of tabular and image-derived features.

Tabular features:
- `bed`, `bath`, `sqft`
- `city_target_mean` — training-set mean price per city, mapped onto val/test (target encoding to avoid 415-way one-hot on a small dataset). Cities not seen in train get the global train mean.

Image features (computed in vanilla numpy / PIL on the single exterior photo):
- `mean_r`, `mean_g`, `mean_b` — mean RGB
- `brightness` — mean of grayscale
- `contrast` — std of grayscale
- `edge_density` — Sobel edge magnitude, mean over the image
- `hist_r_0..hist_r_3`, same for g, b — 4-bin color histograms per channel (12 features)
- 19 image features total

We deliberately exclude `street` (free text, near-unique per house) and `n_citi` (a label encoding of `citi` that has no monotonic meaning, so it would mislead linear and tree models).

## What we are NOT doing
- Not using deep learning, pretrained CNNs, or any image embeddings beyond the 19 hand-computed numpy features.
- Not using R² or any normalized variance metric.
- Not one-hot encoding `citi` (415 levels, would explode dimensionality on a small dataset).
- Not stacking models or building ensembles for the sake of it. We compare three single regressors and report the comparison.
- Not log-transforming price. Range is bounded at $2M, regression in dollars is honest and the metric stays interpretable.

## Story arc for the deck
The deck moves through four plain steps. Each step is one slide block, no setups or reveals.

1. **The question.** Predict a house's price. What's already there: beds, baths, sqft, city. What's missing: the photo. Can the photo close any gap?
2. **The data.** 15,474 SoCal listings. Show the price histogram, a few sample houses with their specs, and the city count.
3. **Specs-only baseline.** Linear regression, decision tree, KNN on beds/baths/sqft/city_target_mean. Report MAE/RMSE/within-thresholds. This is the bar to beat.
4. **Specs + image features.** Re-run all three models with the 19 image features added. Report the same metrics. Show the gap.
5. **Where the photo helps and where it doesn't.** Per-city MAE comparison (does the photo help more in some cities?), and a residual analysis (which houses does the specs-only model miss that the image model catches?).
6. **Bonus clustering.** K-means on image features alone, show that exterior-only clusters separate roughly by price band even without seeing the specs. Hierarchical clustering on cities (using train mean prices and mean image features per city) to show which cities look alike.
7. **Conclusion.** State flatly what improved, by how much, and where the ceiling sits.

## Findings (to be filled in after running)
1. _TBD_
2. _TBD_
3. _TBD_

## Model results, validation set (~2,321 rows)
To be filled in after `analyze.py` runs. Schema:

| Model | Features | MAE | RMSE | SSE | within $50k | within $100k |
|---|---|---|---|---|---|---|
| Constant predictor (train mean) | none | _ | _ | _ | _ | _ |
| Linear regression | specs only | _ | _ | _ | _ | _ |
| Linear regression | specs + image | _ | _ | _ | _ | _ |
| Decision tree (depth tuned on val) | specs only | _ | _ | _ | _ | _ |
| Decision tree (depth tuned on val) | specs + image | _ | _ | _ | _ | _ |
| KNN (k tuned on val) | specs only | _ | _ | _ | _ | _ |
| KNN (k tuned on val) | specs + image | _ | _ | _ | _ | _ |

Final test, model frozen on validation: _TBD_.

## Subgroup robustness (to be filled in)
Per-city MAE for the top 10 cities, specs-only vs specs+image, on validation.

| city | n | specs MAE | specs+image MAE | delta |
|---|---|---|---|---|
| San Diego | _ | _ | _ | _ |
| Los Angeles | _ | _ | _ | _ |
| ... | _ | _ | _ | _ |

## Project layout
```
/home/noah-gallego/Dropbox/Desktop/house-price-4450-final/   # code
  CLAUDE.md
  HousePrice_Final_Project.pptx                              # final deck
  scripts/
    lib_house.py          # shared utilities (loads, splits, metrics, CSUB colors)
    01_inspect.py         # initial shape inspection, summary stats
    02_extract_features.py # compute the 19 image features per house, save parquet
    03_eda.py             # price histogram, sample-houses grid, city counts
    04_baseline.py        # constant predictor + specs-only linear regression
    05_specs_models.py    # specs-only linear, tree, KNN with val-set tuning
    06_image_models.py    # specs+image versions of the same three models
    07_cluster.py         # K-means on image features, hierarchical on cities
    analyze.py            # full final pipeline, writes results/metrics.json
    make_figures.py       # CSUB-styled figures for the deck
    build_pptx.py         # PPTX assembler
  figures/                # fig01..figNN.png (CSUB blue + gold theme)
  results/
    metrics.json          # every model's full metric set

/home/noah-gallego/house-data/                # data (OUTSIDE Dropbox)
  raw/
    socal2.csv            # 15,474 rows
    socal2/socal_pics/    # 15,474 jpgs, one per house
    house-prices-and-images-socal.zip
  clean/
    features.parquet      # tabular + 19 image features per house
    splits.npz            # canonical 65/15/20 indices (random_state=42)
```

## Environment
- Python 3.13.5 (miniconda)
- pandas, numpy, scikit-learn, matplotlib, scipy, pyarrow, Pillow, python-pptx
- Kaggle CLI 2.1.2, credentials at `~/.kaggle/kaggle.json`

## Deliverables
- `HousePrice_Final_Project.pptx` — final deck, CSUB colors (#003594 / #FDB913). Slide structure follows the KNN-Midterm template in tone and format.
- `results/metrics.json` — every model's full metric set
- `figures/figNN.png` — styled figures used in the deck

## Reproduction
```
cd /home/noah-gallego/house-data/raw
kaggle datasets download -d ted8080/house-prices-and-images-socal
unzip house-prices-and-images-socal.zip

cd /home/noah-gallego/Dropbox/Desktop/house-price-4450-final
python3 scripts/02_extract_features.py   # builds features.parquet, ~15 min
python3 -c "import sys; sys.path.insert(0,'scripts'); from lib_house import save_splits; save_splits()"
python3 scripts/analyze.py
python3 scripts/make_figures.py
python3 scripts/build_pptx.py
```

## Status
- [x] Dataset chosen, downloaded, unzipped to `~/house-data/raw/`
- [x] Project folder scaffolded at `~/Dropbox/Desktop/house-price-4450-final/`
- [x] CLAUDE.md written
- [ ] `lib_house.py` (loads, splits, metrics, color theme) written
- [ ] Image feature extraction pass over 15,474 photos
- [ ] Canonical 65/15/20 splits saved
- [ ] Specs-only models fit and evaluated on validation
- [ ] Specs+image models fit and evaluated on validation
- [ ] K-means and hierarchical clustering bonus runs
- [ ] Final test evaluation locked
- [ ] CSUB-styled figures generated
- [ ] PPTX deck assembled

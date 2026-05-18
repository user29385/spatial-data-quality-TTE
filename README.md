# Spatial Data Quality Polluter for TTE

This repository contains data pollution functions used in the paper:

> *Assessing the Impact of Spatial Data Quality on Deep Learning Models for Travel Time Estimation*

The project evaluates how spatial data quality degradation affects Deep Learning models for **Travel Time Estimation (TTE)** using GPS trajectory datasets.

The implemented pollution methods simulate two spatial data quality dimensions from **ISO 19157-1:2023**:

* **Positional Accuracy** — Gaussian noise injection in GPS points
* **Completeness** — Removal of trajectory segments (Trajectory Trimming)

The experiments were conducted using the Porto Taxi dataset.

---

# Repository Structure

```text
.
├── main.py
├── polluter.py
└── README.md
```

---

# Requirements

Install dependencies:

```bash
pip install pandas numpy
```

---

# Dataset Format

The input CSV must contain a `POLYLINE` column with trajectories encoded as JSON.

Example:

```csv
POLYLINE
"[[ -8.618643,41.141412 ],[ -8.618499,41.141376 ]]"
```

---

# Experiments Dataset

The dataset used in the experiments was the [Taxi Trajectory Data](https://www.kaggle.com/datasets/crailtap/taxi-trajectory) dataset, containing a complete year of the trajectories for 442 taxis running in the city of Porto, Portugal.

---

# Running the Polluter

Execute:

```bash
python main.py path/to/dataset.csv
```

Example:

```bash
python main.py train.csv
```

If no path is provided, the default path is "data/train.csv"

---

# Generated Files

| File                   | Description                           |
| ---------------------- | ------------------------------------- |
| `taxi_porto_noisy.csv` | Dataset with positional noise         |
| `completude_ds.csv`    | Dataset with completeness degradation |

---

# Pollution Methods

## Positional Accuracy

Applies Gaussian noise to GPS points using:

```
mod_accuracy(df, pct, sigma)
```

The distortion rate for each trajectory is randomly selected from:

```
{0.0, 0.2, 0.4, 0.6, 0.8}
```

---

## Completeness

Applies trajectory trimming using:

```python
mod_completeness_trimming(
    df,
    pct,
    min_gap_ratio,
    max_gap_ratio
)
```

A continuous segment is removed from either the beginning or end of the trajectory.

---

# Citation

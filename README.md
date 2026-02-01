# Industrial Anomaly Detection using Machine Learning

## Overview
This project implements a production-style machine learning pipeline to detect
rare anomalies in industrial sensor data. The system handles severe class
imbalance using time-aware feature engineering, threshold optimization, and
semi-supervised learning.

---
## Dataset
Due to size constraints, the dataset is not included in this repository.

Please place the following files inside a `data/` directory:
- `train.parquet` (contains features + target)
- `test.parquet` (contains features only)


## Key Highlights
- Time-series–aware feature engineering
- Imbalanced learning with F1-score optimization
- LightGBM with class-weighting
- Semi-supervised pseudo-labeling
- Dual-threshold decision strategy
- Modular, production-ready codebase

---

## Project Structure

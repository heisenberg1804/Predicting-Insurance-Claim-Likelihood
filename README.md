Based on your project report, here's a crisp GitHub README template:

# Predicting Insurance Claim Likelihood Based on Claim Features and Extreme Weather Events

## Overview
A machine learning project that predicts flood insurance claim approval likelihood by integrating NFIP (National Flood Insurance Program) claims data with NOAA storm events data. This project achieves **~93-94% accuracy** and **~0.96 ROC AUC** using ensemble methods.

## Project Highlights
- 📊 **Dataset**: 612,485 NFIP claims (2014-2024) merged with NOAA storm events
- 🎯 **Best Model**: Gradient Boosting with 94% accuracy and 0.96 ROC AUC
- 🔍 **Key Insight**: Core NFIP attributes (building replacement cost, water depth) are primary predictors, while storm data provides valuable incremental context

## Data Sources
1. **NFIP Claims Dataset** - FEMA NFIP Redacted Claims V2
2. **NOAA Storm Events Database** - Historical extreme weather records (2014-2024)

## Methodology

### Data Pipeline
1. **Preprocessing**
   - Filtered valid claims (2014-2024)
   - Removed target leakage features
   - Handled outliers and missing values
   - Feature engineering (building age, policy age, temporal features)

2. **Merging Strategy**
   - Temporal matching: ±3-4 days tolerance
   - Spatial matching: ±2° latitude/longitude range

3. **Modeling**
   - Logistic Regression (baseline)
   - Random Forest
   - **Gradient Boosting** ⭐
   - Multi-Layer Perceptron

### Key Features Identified
- `buildingReplacementCost`
- `waterDepth`
- `contentsPropertyValue`
- `MAGNITUDE_TYPE` (from NOAA)
- `EVENT_TYPE` (from NOAA)

## Results

| Model | Accuracy | ROC AUC | Macro F1 |
|-------|----------|---------|----------|
| Logistic Regression | 77-79% | 0.79-0.83 | 0.70-0.71 |
| Random Forest | 93% | 0.95-0.96 | 0.89-0.90 |
| **Gradient Boosting** | **94%** | **0.96** | **0.90** |
| MLP Neural Network | 90% | 0.93 | 0.86 |

## Repository Structure
```
├── data/
│   ├── nfip/          # NFIP claims data
│   └── noaa/          # NOAA storm events
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   └── modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
└── results/
    ├── figures/
    └── model_outputs/
```

## Key Findings
1. **Tree ensembles outperform** linear and neural network models
2. **NFIP claim attributes dominate** feature importance rankings
3. **Storm data provides marginal gains** for simple models but minimal improvement for ensemble methods
4. **SHAP analysis reveals** building/property features as primary drivers of claim approval

## Future Work
- Explore enhanced merging strategies with relaxed spatial-temporal tolerances
- Implement HistGradientBoostingClassifier for native NaN handling
- Investigate deep learning architectures
- Expand feature engineering for storm-specific signals

## Course
IE 7275 - Data Mining in Engineering, Northeastern University

---
*Note: This project demonstrates the application of machine learning techniques to improve insurance risk assessment by integrating multiple data sources for better predictive accuracy.*

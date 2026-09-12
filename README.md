<div align="center">

#  Football Player Market Value Prediction

### Predicting FIFA player market values with Random Forest Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

*An end-to-end regression pipeline that predicts a player's market value (€) from FIFA attribute data — achieving an R² of 0.988.*

[Overview](#-overview) • [Dataset](#-dataset) • [Methodology](#-methodology) • [Results](#-results) • [Insights](#-key-insights) • [Installation](#-installation)

</div>

---

##  Overview

Football clubs, agents, and analysts routinely need to estimate a player's fair market value — for transfer negotiations, squad valuation, or scouting decisions. Market value is driven by a mix of measurable performance attributes (overall rating, technical skills) and physical traits, making it a strong candidate for a data-driven regression approach.

This project trains a **Random Forest Regression** model on FIFA player attribute data to predict `value_eur`, systematically tuning the number of trees (`n_estimators`) to find the best trade-off between accuracy and computational cost.

**What this project demonstrates:**

| Capability | Implementation |
|---|---|
| Data cleaning | Missing-value removal on an 18.9K-row dataset |
| Feature selection | 14 relevant numerical attributes isolated from the full player schema |
| Leakage-safe evaluation | 80/20 train/test split |
| Hyperparameter tuning | Systematic comparison across 5 forest sizes |
| Model interpretability | Feature importance ranking to explain valuation drivers |
| Diagnostic evaluation | Actual-vs-predicted and residual analysis, beyond headline metrics |

---

##  Dataset

**Source:** [Football Price Prediction — Kaggle](https://www.kaggle.com/datasets/thedevastator/footballpriceprediction)

### Dataset Summary

| Property | Value |
|---|---:|
| Original Rows | 18,945 |
| Rows After Preprocessing | 16,861 |
| Selected Features | 14 |
| Target Variable | `value_eur` |

### Feature Schema

<details open>
<summary><b>Selected Numerical Features</b></summary>

| Feature | Description |
|---|---|
| `age` | Player's age |
| `overall` | Current overall rating |
| `potential` | Projected ceiling rating |
| `height_cm` | Height in centimeters |
| `weight_kg` | Weight in kilograms |
| `weak_foot` | Weak-foot skill rating |
| `skill_moves` | Skill move rating |
| `international_reputation` | International reputation rating |
| `pace` | Pace attribute |
| `shooting` | Shooting attribute |
| `passing` | Passing attribute |
| `dribbling` | Dribbling attribute |
| `defending` | Defending attribute |
| `physic` | Physical attribute |

</details>

**Target:** `value_eur` — the player's market value in euros.

---

##  Data Preprocessing

- Isolated the dataset to the **14 relevant numerical features** above, dropping non-predictive or non-numerical columns
- Removed rows containing missing values, reducing the dataset from **18,945 → 16,861 rows** (a ~11% reduction, retaining the large majority of the data)
- Split the cleaned dataset into **80% training / 20% testing** subsets

---

##  Methodology

### Why Random Forest?

Random Forest Regression was chosen because player valuation is driven by **non-linear interactions** between attributes (e.g., a high `overall` rating combined with high `potential` and youth compounds value in a way a linear model can't easily capture). As an ensemble of decision trees, Random Forest:

- Captures non-linear relationships and feature interactions natively
- Is robust to outliers and doesn't require feature scaling
- Provides built-in feature importance for interpretability

### Hyperparameter Tuning

Several values of `n_estimators` (number of trees) were evaluated to find the point of diminishing returns:

| Trees | R² Score | MAE (€) | RMSE (€) |
|---:|---:|---:|---:|
| 10 | 0.984191 | 150,649.42 | 669,794.14 |
| 100 | 0.986909 | 136,935.41 | 609,523.61 |
| 200 | 0.987470 | 135,557.55 | 596,301.57 |
| **300** | **0.987713** | **134,716.99** | **590,496.91** |
| 500 | 0.987561 | 134,317.84 | 594,148.20 |

**Observation:** performance improves steadily up to **300 trees**, then plateaus — 500 trees actually performs marginally *worse* on this test split while costing more compute. This is a classic Random Forest signature: past a certain ensemble size, additional trees reduce variance only negligibly while training time keeps growing.

### Final Model Configuration

| Parameter | Value |
|---|---:|
| Model | Random Forest Regressor |
| `n_estimators` | 300 |
| `random_state` | 42 |

---

##  Results

### Headline Metrics

<div align="center">

###  R² Score: **0.9877**

</div>

| Metric | Formula | Value | Interpretation |
|---|---|---:|---|
| **R² Score** | 1 − (SS_res / SS_tot) | **0.9877** | The model explains ~98.8% of the variance in player market value |
| **MAE** | (1/n)·Σ\|yᵢ − ŷᵢ\| | **€134,716.99** | On average, predictions are off by ~€135K |
| **RMSE** | √[(1/n)·Σ(yᵢ − ŷᵢ)²] | **€590,496.91** | Typical error, penalizing large misses more heavily than MAE |

> The gap between MAE and RMSE suggests the model has some larger errors on a subset of predictions — likely high-value outlier players (superstars), whose valuations are driven by market and reputation factors beyond the attributes in this dataset.

---

##  Visualizations

### Actual vs. Predicted Value

Predictions cluster tightly around the perfect-prediction diagonal, visually confirming the high R² score and showing the model generalizes well across the value range.

<p align="center">
  <img src="Images/actual_vs_predicted.png" width="700">
</p>

---

### Feature Importance

Ranks which attributes the Random Forest relied on most heavily when predicting market value — key for explaining *why* the model values a player the way it does, not just *what* it predicts.

<p align="center">
  <img src="Images/feature_importance.png" width="700">
</p>

---

### Residual Plot

Residuals are randomly scattered around zero with no strong systematic pattern — evidence that the model isn't systematically over- or under-predicting for any particular value range, and that it has captured the underlying structure in the data rather than a biased approximation of it.

<p align="center">
  <img src="Images/residual_plot.png" width="700">
</p>

---

##  Key Insights

- **`overall` rating is the dominant predictor** of market value by a wide margin — unsurprising, since it's the single most direct summary of a player's current ability.
- **`potential` is the second most influential feature**, indicating the market prices in *future* development, not just current performance — younger players with high ceilings command a premium.
- **Technical attributes** (shooting, passing, dribbling, defending) contribute meaningfully but with far smaller individual impact than `overall` and `potential`.
- **Forest size scaling**: performance improved consistently up to **300 trees**, after which additional trees yielded only marginal (or slightly negative) returns — confirming 300 as the practical sweet spot for this dataset.

---

##  Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn (Random Forest Regressor) |
| Environment | Jupyter Notebook |

---

##  Project Structure

```
Football-Player-Market-Value-Prediction/
│
├── Images/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── residual_plot.png
├── data/
│   └── fifa_players.csv
├── notebooks/
│   └── market_value_prediction.ipynb
├── README.md
└── requirements.txt
```

---

##  Installation

```bash
# 1. Clone the repository
git clone https://github.com/khaled-amireh/Football-Player-Market-Value-Prediction.git
cd Football-Player-Market-Value-Prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/market_value_prediction.ipynb
```

---

## ⚠️ Limitations

- The model was evaluated on a single train/test split rather than cross-validated — reported metrics may vary somewhat across different splits.
- Market value for elite/superstar players can be influenced by factors outside this dataset (brand value, marketability, contract situation), which likely explains the larger residuals noted for high-value outliers.
- Only numerical FIFA attributes were used; categorical context (club, league, nationality, position) was not incorporated and could add predictive signal.

---

## 🚀 Future Improvements

- [ ] K-Fold Cross-Validation for a more robust performance estimate
- [ ] Incorporate categorical features (position, league, nationality) via encoding
- [ ] Hyperparameter tuning beyond `n_estimators` (max_depth, min_samples_split)
- [ ] Benchmark against Gradient Boosting (XGBoost/LightGBM)
- [ ] SHAP value analysis for deeper, per-prediction interpretability

---

## ✅ Conclusion

Random Forest Regression achieved excellent predictive performance with an **R² score of 0.9877**, explaining nearly 99% of the variance in player market value.

The results confirm that player ratings — particularly **overall** and **potential** — are the dominant drivers of valuation. Hyperparameter tuning showed that **300 decision trees** provides the best balance between predictive performance and computational cost for this dataset, with diminishing (and occasionally negative) returns beyond that point.

---

## 👤 Author

**Khaled Amireh**
[GitHub](https://github.com/khaled-amireh)

---

<div align="center">

*If you found this project useful, consider giving it a ⭐ on GitHub.*

</div>

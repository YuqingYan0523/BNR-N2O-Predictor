# BNR-N2O Tier Predictor
Authors: [Yuqing Yan](https://github.com/YuqingYan0523) and [Junjie Zhu](https://github.com/starfriend10) @Princeton University WET Lab led by Z. Jason Ren

## Project Overview
This project predicts nitrous oxide (N₂O) emission factor (EF) levels from full-scale biological nitrogen removal (BNR) systems using machine learning. The model is trained on literature-derived metadata, which includes both numeric features (e.g., dissolved oxygen) and categorical features (e.g., process configuration). Emission factor levels were grouped into four tiers: *High*, *Medium-High*, *Medium-Low*, and *Low* using Gaussian Mixture Model clustering of reported EF values.

To ensure robust and unbiased model training, data were split by study (not individual observation) to avoid leakage, and stratified to preserve the distribution of both inputs and emission tiers. Given the limited and imbalanced dataset, 2000 random seeds were tested, yielding 30 valid splits that met these criteria. These were used for model selection and evaluation.

Multiple classification models (e.g., Gradient Boosting, Random Forest, CatBoost, MLP) and regression models were tested, incorporating class weights to address label imbalance. Final model selection was based on performance consistency across splits. Feature selection was performed using permutation importance and SHAP values, followed by hyperparameter tuning via grid search with cross-validation. Eventually, an optimal Gradient Boosting model with an overall accuracy of 0.52 was developed with 0.6-0.7 AUCs across the four tiers was develped.

## Environment
Please make sure you have the following packages installed:

- catboost: 1.2.8
- joblib: 1.5.1
- matplotlib: 3.10.0
- numpy: 2.0.2
- pandas: 2.2.2
- scipy: 1.16.0
- shap: 0.48.0
- sklearn: 1.6.1
- tensorflow: 2.18.0
- tqdm: 4.67.1
- xgboost: 3.0.2

## Predictors
The predictor features can be divided into two groups: Numeric and Categorical, detailed as following:

| Feature        | Type      | Description          |
|----------------|-----------|----------------------|
| Volume         | Numeric   | Reactor Volume (m3)    |
| NLR            | Numeric   | Nitrogen Loading Rate (kg/d)   |
| Eff_NH4        | Numeric   | Effluent NH4+ Concentration (mg/L)    |
| Inf_TN         | Numeric   | Inffluent Total Nitrogen (mg/L)    |
| HRT            | Numeric   | Hydrualic Retention Time (h)    |
| DO             | Numeric   | Dissolved Oxygen (mg/L)    |
| Temp           | Numeric   | Temperature (K)    |
| Operation      | Categorical | BNR Process Type: one of 'Multi-stage AS' (e.g. AAO, MLE, AO), 'Single-stage AS' (e.g. CAS), 'OD' (Oxidation Ditch) or 'SBR' (Sequencing Batch Reactor)  |
| Condition      | Categorical | Aeration Conditions: one of 'Undifferentiated' (either overall emission for both aerated and non-aerated stages, or unspecified condition), 'Aerated', 'Non-Aerated'|

## How to use the model
1. Download or clone this repo
```bash
git clone https://github.com/YuqingYan0523/BNR-N2O-Tier-Predictor.git
cd BNR-N2O-Tier-Predictor
```
2. Ensure your input data matches the format

Prepare a single observation with the same categorical and numeric features, i.e. your inquiry facility information, used during training.

3. Run the prediction code
   
Below is an example inference script:
```python
import joblib
import pandas as pd

# ========================================= #
# 1 Load model, label encoder and scaler    #
# ========================================= #
model = joblib.load("n2o_model.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# ========================================= #
# 2 Create input sample                     #
# ========================================= #

## 2.1 define feature types
categorical_features = ['Feature_Cat1', 'Feature_Cat2']  # update with your actual categorical column names
numeric_features = ['Feature_Num1', 'Feature_Num2']      # update with your actual numeric column names
model_features = categorical_features + numeric_features # total feature order used in training

## 2.2 fill in your data
sample = pd.DataFrame({
    'Volume': [10.2], # unit: m3
    'NLR': [210], # unit: kg/d
    'Eff_NH4': [19.0], # unit: mg/L
    'Inf_TN': [85], # unit: mg/L
    'HRT': [24], # unit: h
    'DO': [0.5], # unit: mg/L
    'Temp': [298], # unit: K
    'Condition_Aerated': [0], # 0: no, 1: yes
    'Condition_Non-Aerated': [1], # 0: no, 1: yes
    'Condition_Undifferentiated': [0], # 0: no, 1: yes
    'Operation_Multi-stage AS': [0], # 0: no, 1: yes
    'Operation_OD': [0], # 0: no, 1: yes
    'Operation_SBR': [0], # 0: no, 1: yes
    'Operation_Single-stage AS': [1] # 0: no, 1: yes
})

## 2.3 preprocessing
sample_numeric_scaled = pd.DataFrame(
    scaler.transform(sample[numeric_features]),
    columns=numeric_features
)
sample_final = pd.concat([sample[categorical_features].reset_index(drop=True),
                          sample_numeric_scaled.reset_index(drop=True)], axis=1)
sample_final = sample_final[model_features]

# ========================================= #
# 3 Tier Prediction                         #
# ========================================= #
encoded_pred = model.predict(sample_final)
pred_label = le.inverse_transform(encoded_pred)

# Tier Interpretation:
# High: > 3.89%, Medium-High: 0.29 - 3.89%, Medium-Low: 0.0088 - 0.29%, and Low: < 0.0088%; unit: % N2O-N/N removal
print("Predicted EF Level:", pred_label[0])
```
**If you use this model for publication, please cite**

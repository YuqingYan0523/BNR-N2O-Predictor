import joblib
import pandas as pd

# ========================================= #
# 1 Load model, label encoder and scaler    #
# ========================================= #
model = joblib.load("src/n2o_model.pkl") # change path if necessary!
le = joblib.load("src/label_encoder.pkl")
scaler = joblib.load("src/minmax_scaler.pkl")

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

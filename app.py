import streamlit as st
import joblib
import pandas as pd

# Load models/encoders
best_svc = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")
le_country = joblib.load("country_encoder.pkl")
le_used_app = joblib.load("used_app_encoder.pkl")
ohe = joblib.load("onehot_encoder.pkl")
le_class = joblib.load("class_encoder.pkl")

st.title("🧠 Autism Spectrum Disorder Prediction App")

st.write("Fill in the details below and click **Predict** to check the probability of Autism Spectrum Disorder.")

# Collect A1–A10 scores
scores = {}
for i in range(1, 10 + 1):
    scores[f"A{i}_Score"] = st.selectbox(f"A{i}_Score", [0, 1], index=0)

# Other inputs
age = st.number_input("Age", min_value=1, max_value=100, value=10)
gender = st.selectbox("Gender", ["male", "female"])
jaundice = st.selectbox("History of jaundice?", ["no", "yes"])
autism = st.selectbox("Family history of autism?", ["no", "yes"])
country = st.text_input("Country of residence", "India")
used_app_before = st.selectbox("Used app before?", ["no", "yes"])

# Convert inputs to dict
user_input = {
    **scores,
    "age": age,
    "gender": gender,
    "jaundice": jaundice,
    "autism": autism,
    "country_of_res": country,
    "used_app_before": used_app_before,
}

# Prediction button
if st.button("Predict"):
    new_df = pd.DataFrame([user_input])

    # One-hot encode categorical columns
    categorical_cols_ohe_predict = ["gender", "jaundice", "autism"]
    encoded_data_ohe_predict = ohe.transform(new_df[categorical_cols_ohe_predict])
    ohe_feature_names_predict = ohe.get_feature_names_out(categorical_cols_ohe_predict)
    encoded_df_ohe_predict = pd.DataFrame(
        encoded_data_ohe_predict,
        columns=ohe_feature_names_predict,
        index=new_df.index,
    )
    new_df = pd.concat([new_df, encoded_df_ohe_predict], axis=1)
    new_df.drop(columns=categorical_cols_ohe_predict, inplace=True)

    # Label encode
    new_df["country_of_res"] = le_country.transform(new_df["country_of_res"])
    new_df["used_app_before"] = le_used_app.transform(new_df["used_app_before"])

    # ✅ Ensure feature order matches training
    try:
        expected_features = scaler.feature_names_in_   # sklearn >=1.0
    except AttributeError:
        # If not available, define manually (must match training pipeline!)
        expected_features = [
            "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
            "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
            "age","country_of_res","used_app_before",
            # add all OHE-generated columns
            "gender_female","gender_male",
            "jaundice_no","jaundice_yes",
            "autism_no","autism_yes"
        ]

    # Reindex new_df to match training features
    new_df = new_df.reindex(columns=expected_features, fill_value=0)

    # Scale
    new_scaled = scaler.transform(new_df)

    # Predict
    prediction = best_svc.predict(new_scaled)[0]
    predicted_class_label = le_class.inverse_transform([prediction])[0]

    st.subheader(f"✅ Predicted Class: {predicted_class_label}")

    # Show probabilities
    if hasattr(best_svc, "predict_proba"):
        probabilities = best_svc.predict_proba(new_scaled)[0]
        prob_df = pd.DataFrame(
            {
                "Class": le_class.inverse_transform(best_svc.classes_),
                "Probability": probabilities,
            }
        )
        st.write("📊 Prediction Probabilities")
        st.dataframe(prob_df)

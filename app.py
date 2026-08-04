import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Food Delivery ETA Predictor", page_icon="🛵")

model = joblib.load("delivery_time_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")
# Whether this model was trained on scaled input (True for Linear/Ridge,
# False for tree models like Random Forest / Gradient Boosting / XGBoost).
uses_scaled_input = joblib.load("uses_scaled_input.pkl")

st.title("🛵 Smart Food Delivery ETA Predictor")
st.caption("Estimate delivery time from distance, weather, traffic, and courier details.")

distance = st.slider("Distance (km)", 0.5, 30.0, 5.0)
prep_time = st.slider("Preparation Time (min)", 5, 60, 15)
experience = st.slider("Courier Experience (years)", 0.0, 15.0, 2.0)

weather = st.selectbox("Weather", ["Clear", "Windy", "Foggy", "Rainy", "Cloudy", "Snowy"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

if st.button("Predict Delivery Time"):
    # Start every one-hot column at 0
    row = {c: 0 for c in model_columns}

    # Fill in the numeric features (only set if the model actually has them)
    for col_name, value in [
        ("Distance_km", distance),
        ("Preparation_Time_min", prep_time),
        ("Courier_Experience_yrs", experience),
    ]:
        if col_name in row:
            row[col_name] = value

    # Flip the matching one-hot dummy column to 1 for each categorical choice
    for prefix, value in [
        ("Weather", weather),
        ("Traffic_Level", traffic),
        ("Vehicle_Type", vehicle),
        ("Time_of_Day", time_of_day),
    ]:
        dummy_col = f"{prefix}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1
        # Note: if this is the category that got dropped by drop_first=True,
        # leaving all dummies at 0 for that feature IS the correct encoding.

    input_df = pd.DataFrame([row])[model_columns]

    # Apply the same scaling used during training, only if this model needs it
    if uses_scaled_input:
        model_input = scaler.transform(input_df)
    else:
        model_input = input_df

    prediction = model.predict(model_input)[0]

    st.success(f"Estimated Delivery Time: **{prediction:.1f} minutes**")

    with st.expander("See exact input sent to the model"):
        st.dataframe(input_df)

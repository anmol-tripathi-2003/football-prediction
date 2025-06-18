import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the trained model and label encoders
model = joblib.load("model.pkl")
team_encoder = joblib.load("team_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
day_encoder = joblib.load("day_encoder.pkl")

# Load match data
matches = pd.read_csv("matches.csv")

# Team logos dictionary (add as needed)
team_logos = {
    "Arsenal": "logos/arsenal.png",
    "Chelsea": "logos/chelsea.png",
    "Liverpool": "logos/liverpool.png",
    "Manchester City": "logos/man_city.png",
    "Manchester Utd": "logos/man_utd.png",
    "Tottenham": "logos/tottenham.png",
    # Add more logos here
}

# Streamlit app title
st.markdown("<h1 style='text-align: center;'>⚽ Fantasy Football Match Winner Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input: Select Team and Opponent
team = st.selectbox("Select Your Team", [""] + sorted(matches["team"].unique()))
opponent = st.selectbox("Select Opponent", [""] + sorted(matches["opponent"].unique()))
venue = st.selectbox("Select Match Venue", [""] + sorted(matches["venue"].unique()))

# Input: Match Time and Day
time = st.slider("Match Start Time (24-hour format)", 0, 23, 15)
day = st.selectbox("Match Day", sorted(matches["day"].unique()))

st.markdown("---")

# Prediction logic
if team and opponent and venue:
    # Show team logo (optional)
    if team in team_logos:
        st.image(team_logos[team], width=100)

    # Prepare input data
    input_data = pd.DataFrame({
        "team": [team],
        "opponent": [opponent],
        "venue": [venue],
        "time": [time],
        "day": [day]
    })

    # Encode inputs
    input_data["team"] = team_encoder.transform(input_data["team"])
    input_data["opponent"] = team_encoder.transform(input_data["opponent"])
    input_data["venue"] = venue_encoder.transform(input_data["venue"])
    input_data["day"] = day_encoder.transform(input_data["day"])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Output result
    if prediction == 1:
        st.success("✅ Prediction: Win")
    else:
        st.error("❌ Prediction: Loss")
else:
    st.warning("Please select both teams and venue to see prediction.")

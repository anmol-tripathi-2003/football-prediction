import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load data
matches = pd.read_csv("matches.csv", index_col=0)
matches['date'] = pd.to_datetime(matches['date'])

# Preprocessing
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(':.+', '', regex=True).astype('int')
matches['day_code'] = matches['date'].dt.dayofweek
matches['target'] = (matches['result'] == 'W').astype('int')

# Train the model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] >= '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train["target"])

# --- Streamlit App ---
st.title("⚽ Football Match Winner Predictor")

# 1. Select team
team = st.selectbox("Select Your Team", sorted(matches["team"].unique()))

# Filter dataset for selected team
team_matches = matches[matches["team"] == team]

# 2. Select venue
venue = st.selectbox("Select Match Venue", sorted(team_matches["venue"].unique()))
venue_code = team_matches[team_matches["venue"] == venue]["venue_code"].iloc[0]

# 3. Select opponent
opponent = st.selectbox("Select Opponent", sorted(team_matches["opponent"].unique()))
opp_code = team_matches[team_matches["opponent"] == opponent]["opp_code"].iloc[0]

# 4. Match time
hour = st.slider("Match Start Time (24-hour format)", 0, 23, 20)

# 5. Match day
day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
           4: "Friday", 5: "Saturday", 6: "Sunday"}
day_name = st.selectbox("Match Day", list(day_map.values()))
day_code = [k for k, v in day_map.items() if v == day_name][0]

# Predict
input_data = pd.DataFrame({
    "venue_code": [venue_code],
    "opp_code": [opp_code],
    "hour": [hour],
    "day_code": [day_code]
})

prediction = rf.predict(input_data)[0]

# Display result
result = "✅ Win" if prediction == 1 else "❌ Not Win"
st.subheader(f"Prediction for **{team}** vs **{opponent}**: {result}")

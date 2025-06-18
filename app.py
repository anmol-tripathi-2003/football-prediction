import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import datetime

st.title("⚽ Football Match Winner Predictor")

# Load data
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(':.+', '', regex=True).astype('int')
matches['day_code'] = matches['date'].dt.dayofweek
matches['target'] = (matches['result'] == 'W').astype('int')

# --- TEAM SELECTION ---
team = st.selectbox("Select Your Team", matches["team"].unique())
opponent = st.selectbox("Select Opponent", matches["opponent"].unique())
venue = st.selectbox("Select Match Venue", matches["venue"].unique())
hour = st.slider("Match Start Time (24-hour format)", 0, 23)
day = st.selectbox("Match Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# --- TEAM LOGO DISPLAY ---
team_logos = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Manchester City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Manchester Utd": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg"
}
if team in team_logos:
    st.image(team_logos[team], width=100)

# --- RECENT FORM ---
st.subheader(f"{team}'s Recent Form (Last 5 Matches)")
recent = matches[matches["team"] == team].sort_values("date").tail(5)
form = recent[["date", "result"]].copy()
form["result_num"] = recent["target"]
form = form.set_index("date")
st.line_chart(form["result_num"])

# --- MODEL TRAINING ---
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
rf.fit(train[predictors], train["target"])

# --- MAKE PREDICTION ---
input_data = pd.DataFrame({
    "venue_code": [matches[matches["venue"] == venue]["venue_code"].values[0]],
    "opp_code": [matches[matches["opponent"] == opponent]["opp_code"].values[0]],
    "hour": [hour],
    "day_code": [["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)]
})

pred = rf.predict(input_data)[0]
prob = rf.predict_proba(input_data)[0][1]

st.subheader(f"Prediction for {team} vs {opponent}:")
if pred == 1:
    st.success(f"✅ Win (Probability: {round(prob*100, 2)}%)")
else:
    st.error(f"❌ Not a Win (Probability: {round(prob*100, 2)}%)")

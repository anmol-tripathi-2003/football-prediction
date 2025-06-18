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

# TEAM SELECTION
team = st.selectbox("Select Your Team", matches["team"].unique())
opponent = st.selectbox("Select Opponent", matches["opponent"].unique())
venue = st.selectbox("Select Match Venue", matches["venue"].unique())
hour = st.slider("Match Start Time (24-hour format)", 0, 23)
day = st.selectbox("Match Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# TEAM LOGO DISPLAY
team_logos = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_logo.svg",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/e/e6/Brentford_FC_crest.svg",
    "Brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/0/02/Burnley_FC_badge.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo.svg",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/e0/Fulham_FC_%28shield%29.svg",
    "Leeds United": "https://upload.wikimedia.org/wikipedia/en/0/0c/Leeds_United_Logo.svg",
    "Leicester City": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Luton Town": "https://upload.wikimedia.org/wikipedia/en/8/81/Luton_Town_F.C._logo.svg",
    "Manchester City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Manchester Utd": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Newcastle Utd": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Nottingham Forest": "https://upload.wikimedia.org/wikipedia/en/5/53/Nottingham_Forest_F.C._logo.svg",
    "Sheffield Utd": "https://upload.wikimedia.org/wikipedia/en/3/3e/Sheffield_United_FC_logo.svg",
    "Southampton": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "West Ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "Wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg"
}

if team in team_logos:
    st.image(team_logos[team], width=100)

# RECENT FORM 
st.subheader(f"{team}'s Recent Form (Last 5 Matches)")
recent = matches[matches["team"] == team].sort_values("date").tail(5)
form = recent[["date", "result"]].copy()
form["result_num"] = recent["target"]
form = form.set_index("date")
st.line_chart(form["result_num"])

# MODEL TRAINING 
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
rf.fit(train[predictors], train["target"])

# MAKE PREDICTION
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

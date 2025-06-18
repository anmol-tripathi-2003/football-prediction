
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load data
matches = pd.read_csv("matches.csv", index_col=0)
matches['date'] = pd.to_datetime(matches['date'])
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(':.+','', regex=True).astype(int)
matches['day_code'] = matches['date'].dt.dayofweek
matches['target'] = (matches['result']=='W').astype(int)

# Train model
predictors = ["venue_code", "opp_code", "hour", "day_code"]
train = matches[matches["date"] < '2022-01-01']
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# UI
st.title("⚽ Football Match Winner Predictor")

venue = st.selectbox("Select Match Venue", matches['venue'].unique())
opponent = st.selectbox("Select Opponent", matches['opponent'].unique())
hour = st.slider("Match Start Time (24-hour format)", 0, 23, 15)
day = st.selectbox("Match Day", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}

# Convert user input to codes
venue_code = matches['venue'].astype('category').cat.categories.get_loc(venue)
opp_code = matches['opponent'].astype('category').cat.categories.get_loc(opponent)
day_code = day_map[day]

# Predict
input_data = pd.DataFrame([[venue_code, opp_code, hour, day_code]], columns=predictors)
pred = rf.predict(input_data)[0]
result = "✅ Win" if pred == 1 else "❌ Not Win"
st.subheader(f"Prediction: {result}")

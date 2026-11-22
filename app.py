import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model/ipl_win_probability_model.pkl")

model = load_model()

teams = [
    "Chennai Super Kings",
    "Deccan Chargers",
    "Delhi Capitals",
    "Gujarat Lions",
    "Kings XI Punjab",
    "Kochi Tuskers Kerala",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Pune Warriors",
    "Rajasthan Royals",
    "Rising Pune Supergiants",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]

team_mapping = {
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants"
}

st.title("IPL Win Probability Predictor")
st.caption("Predict the chasing team's chances of winning based on the current match situation.")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting team", teams)

with col2:
    bowling_team = st.selectbox(
        "Bowling team",
        [team for team in teams if team != batting_team]
    )

target = st.number_input("Target score", min_value=1, max_value=400, value=180)
current_score = st.number_input("Current score", min_value=0, max_value=400, value=100)
overs = st.number_input("Overs completed", min_value=0.0, max_value=19.5, value=12.3, step=0.1, format="%.1f")
wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=9, value=3)

if st.button("Predict Win Probability", type="primary", use_container_width=True):
    completed_overs = int(overs)
    balls_in_over = round((overs - completed_overs) * 10)

    if balls_in_over > 5 or (completed_overs == 19 and balls_in_over > 5):
        st.error("Invalid overs format. Use values such as 12.3 or 19.5.")
    elif current_score >= target:
        st.error("The chasing team has already reached the target.")
    else:
        legal_balls = completed_overs * 6 + balls_in_over

        if legal_balls >= 120:
            st.error("The innings has already ended.")
        else:
            runs_left = target - current_score
            balls_left = 120 - legal_balls
            wickets_remaining = 10 - wickets_lost
            current_run_rate = (
                current_score * 6 / legal_balls if legal_balls > 0 else 0
            )

            input_data = pd.DataFrame([{
                "batting_team_clean": team_mapping.get(batting_team, batting_team),
                "bowling_team_clean": team_mapping.get(bowling_team, bowling_team),
                "target": target,
                "current_score": current_score,
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets_remaining": wickets_remaining,
                "current_run_rate": current_run_rate
            }])

            probability = float(model.predict_proba(input_data)[0][1])

            st.subheader("Win Probability")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Chasing team", f"{probability * 100:.2f}%")

            with result_col2:
                st.metric("Defending team", f"{(1 - probability) * 100:.2f}%")

            st.progress(probability)

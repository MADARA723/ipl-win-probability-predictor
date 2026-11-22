import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("model/ipl_win_probability_model.pkl")

    sample = pd.DataFrame([{
        "batting_team_clean": "Mumbai Indians",
        "bowling_team_clean": "Chennai Super Kings",
        "target": 180,
        "current_score": 100,
        "runs_left": 80,
        "balls_left": 45,
        "wickets_remaining": 7,
        "current_run_rate": 8.0
    }])

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0][1] <= 1
    assert abs(probabilities[0].sum() - 1) < 1e-6

# IPL Win Probability Predictor

A machine learning project that predicts the probability of a chasing team winning an IPL match based on the current match situation.

## Features
- Predicts chasing and defending team win probabilities.
- Uses match state: target, current score, overs, and wickets lost.
- Interactive Streamlit interface.
- Trained using historical IPL match data.

## Machine Learning
- Gradient Boosting Classifier
- Scikit-learn Pipeline
- One-hot encoding for teams
- Match-state numerical features

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load("app/random_forest_model.pkl")

# Define the features used by the model (must match training)
features = [
    'session_duration', 'cart_value', 'price_per_item', 'intent_score',
    'total_spent', 'avg_spent', 'email_clicked', 'email_sent',
    'completed_from_email', 'num_interactions', 'num_items_in_cart',
    'num_transactions', 'unique_events', 'unique_referrals',
    'day_of_week_weekend', 'day_of_week_weekday', 'offer_type_bogo',
    'unique_devices', 'time_of_day_evening', 'offer_type_fixed'
]

# Define threshold for classifying high abandonment risk
threshold = 0.65

def predict_prob(input_dict):
    """
    Predicts the probability of class 1 (cart abandonment).
    Input: input_dict (dict of features)
    Output: float probability
    """
    input_array = np.array([input_dict[feat] for feat in features]).reshape(1, -1)
    prob = model.predict_proba(input_array)[:, 1][0]
    return prob

def predict_class(prob):
    """
    Predicts class based on threshold.
    """
    return int(prob >= threshold)

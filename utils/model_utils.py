import os

def save_model(model, model_name):
    os.makedirs('models', exist_ok=True)
    if model_name == "xgboost":
        model.save_model(f"models/{model_name}_model.json")
    else:
        import joblib
        joblib.dump(model, f"models/{model_name}_model.pkl")

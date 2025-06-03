from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_model(x_train, y_train, model_type="random_forest"):
    if model_type == "random_forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
    elif model_type == "xgboost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        model = XGBClassifier(eval_metric='logloss', random_state=42)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)
    return grid_search
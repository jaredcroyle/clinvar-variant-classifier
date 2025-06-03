import os
from src.data_preprocessing import load_and_clean_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.shap_visualization import generate_shap_plots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from utils import report_generator
from utils.model_utils import save_model


def main():
    # load and process data
    df = load_and_clean_data("data/variant_summary.txt.gz")

    X = df.drop(columns=["label", "Type"])
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    os.makedirs("report", exist_ok=True)
    
    x_test.to_csv("data/x_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    models_dict = {}

    for model_name in ["random_forest", "xgboost"]:
        print(f"\nTraining {model_name}...")
        grid_search = train_model(x_train, y_train, model_type=model_name)
        print(f"Best Params for {model_name}:", grid_search.best_params_)
        print(f"Best CV Score for {model_name}:", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        
        save_model(best_model, model_name)
        
        if model_name == "xgboost":
            generate_shap_plots()

        # evaluate and save plots with unique names per model
        evaluate_model(best_model, x_test, y_test)
        report_generator.save_confusion_matrix(
            best_model, x_test, y_test, f"report/confusion_matrix_{model_name}.png")
        report_generator.save_roc_curve(
            best_model, x_test, y_test, f"report/roc_curve_{model_name}.png")
        report_generator.save_pr_curve(
            best_model, x_test, y_test, f"report/pr_curve_{model_name}.png")

        if model_name == "random_forest":
            report_generator.save_tree_plot(
                best_model, feature_names=X.columns, out_path=f"report/decision_tree_{model_name}.png")

        # calculating predictions and probabilities/scores
        y_pred = best_model.predict(x_test)
        y_proba = None
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(x_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_proba = best_model.decision_function(x_test)

        # calculating metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        models_dict[model_name] = {
            "model": best_model,
            "x_test": x_test,
            "y_test": y_test,
            "accuracy": round(accuracy, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "feature_names": X.columns if model_name == "random_forest" else None
        }

    report_generator.generate_html_report(
        models_dict=models_dict,
        output_dir="report",
        template_file="report_template.html"
    )

    print("HTML reports saved to report/")

if __name__ == "__main__":
    main()
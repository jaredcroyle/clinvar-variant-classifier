import os
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def save_confusion_matrix(model, x_test, y_test, out_path):
    disp = ConfusionMatrixDisplay.from_estimator(
        model, x_test, y_test, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_roc_curve(model, x_test, y_test, out_path):
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_pr_curve(model, x_test, y_test, out_path):
    PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_tree_plot(model, feature_names, out_path):
    from sklearn.tree import plot_tree
    estimator = model.estimators_[0]  
    plt.figure(figsize=(20, 10))
    plot_tree(
        estimator,
        filled=True,
        feature_names=feature_names,
        class_names=model.classes_.astype(str),
        rounded=True,
        max_depth=3
    )
    plt.title("Random Forest Decision Tree")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def generate_html_report(
    models_dict,  
    output_dir="report",
    template_file="report_template.html"
):
    os.makedirs(output_dir, exist_ok=True)

    models_report = {}

    for model_name, model_data in models_dict.items():
        model = model_data["model"]
        x_test = model_data["x_test"]
        y_test = model_data["y_test"]
        feature_names = model_data.get("feature_names", None)

        # computing predictions and metrics
        y_pred = model.predict(x_test)
        try:
            y_probs = model.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_probs)
        except Exception:
            roc_auc = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # defining file paths
        cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
        pr_path = os.path.join(output_dir, f"{model_name}_pr_curve.png")
        tree_path = os.path.join(output_dir, f"{model_name}_decision_tree.png")

        # saving plots
        save_confusion_matrix(model, x_test, y_test, cm_path)
        save_roc_curve(model, x_test, y_test, roc_path)
        save_pr_curve(model, x_test, y_test, pr_path)

        if hasattr(model, "estimators_") and (feature_names is not None):
            save_tree_plot(model, feature_names, tree_path)
            tree_path_to_use = tree_path
        else:
            tree_path_to_use = None

        models_report[model_name] = {
            "accuracy": f"{accuracy:.3f}",
            "roc_auc": f"{roc_auc:.3f}" if roc_auc is not None else None,
            "precision": f"{precision:.3f}",
            "recall": f"{recall:.3f}",
            "f1_score": f"{f1:.3f}",
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "pr_curve": pr_path,
            "decision_tree": tree_path_to_use
        }

    # loading Jinja2 template
    env = Environment(loader=FileSystemLoader(searchpath="templates"))
    try:
        template = env.get_template(template_file)
    except TemplateNotFound:
        print(f"Template '{template_file}' not found in 'templates/' directory.")
        return

    html_content = template.render(models_dict=models_report)

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"Report generated: {report_path}")

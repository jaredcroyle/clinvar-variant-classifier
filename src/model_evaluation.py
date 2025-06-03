from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.3f}")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Pathogenic"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

def plot_roc_curve(model, x_test, y_test):
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title("ROC Curve")

def plot_pr_curve(model, x_test, y_test):
    PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
    plt.title("Precision-Recall Curve")

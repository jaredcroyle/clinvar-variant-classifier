import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.3f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Pathogenic"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

def plot_sample_tree(model, feature_names, max_depth=3):
    estimator = model.estimators_[0]
    class_names = [str(cls) for cls in model.classes_]

    plt.figure(figsize=(20, 10))
    plot_tree(estimator,
              filled=True,
              feature_names=feature_names,
              class_names=class_names,
              rounded=True,
              max_depth=max_depth)
    plt.title("Sample Decision Tree from Random Forest")

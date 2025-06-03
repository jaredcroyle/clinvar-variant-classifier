import xgboost as xgb
import shap
import pandas as pd
import matplotlib.pyplot as plt

def generate_shap_plots(model_path="models/xgboost_model.json", x_test_path="data/x_test.csv"):
    # model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # test data
    X_test = pd.read_csv(x_test_path)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("report/shap_summary_plot.png", bbox_inches='tight')
    plt.close()

    # save SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("report/shap_bar_plot.png", bbox_inches='tight')
    plt.close()

    print("SHAP plots saved to report/")
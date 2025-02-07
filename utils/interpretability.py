import os
import shap
import matplotlib.pyplot as plt

from loguru import logger


class Interpretability:
    def __init__(self) -> None:
        pass

    def __call__(self, output_path, train, test, model, label_column='label', n_train_samples=100) -> None:
        x_train = train.drop(columns=[label_column])
        x_test = test.drop(columns=[label_column])

        # sample dataset for faster runtimes
        x_train = shap.sample(x_train, nsamples=n_train_samples)

        pred_function = model.predict_proba
        explainer = shap.KernelExplainer(lambda x: pred_function(x)[:, 1], x_train)
        shap_values = explainer.shap_values(x_test)

        plt.figure()
        shap.summary_plot(shap_values, x_test, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

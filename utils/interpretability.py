import os
import shap
import matplotlib.pyplot as plt

from loguru import logger


class Interpretability:
    def __init__(self, config) -> None:
        pass

    def __call__(self, dir, train, test, model) -> None:
        y_train = train['label']
        x_train = train.drop(columns=['label'])
        y_test = test['label']
        x_test = test.drop(columns=['label'])

        # sample dataset for faster runtimes
        x_train = shap.sample(x_train, nsamples=50)
        x_test = shap.sample(x_test, nsamples=50)

        pred_function = model.predict_proba
        explainer = shap.KernelExplainer(lambda x: pred_function(x)[:, 1], x_train)
        shap_values = explainer.shap_values(x_test)

        shap.summary_plot(shap_values, x_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, 'KernelSHAP.png'), dpi=300)
        plt.clf()

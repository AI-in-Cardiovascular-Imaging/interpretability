import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from roc_utils import plot_roc, plot_roc_bootstrap, compute_roc, plot_mean_roc

import os
from os.path import join
import warnings
import pickle

from utils.roc_comparison import delong_roc_test

from utils.interpretability import Interpretability
from utils.plot import plot_confusion_matrix


def main():
    warnings.filterwarnings("ignore")
    color_palette = matplotlib.colormaps.get_cmap('tab10').colors

    for outcome in ["Cardiac", "Vasc"]:
        path = f"/home/aici/Projects/TAVI_TechFail/V02/{outcome}/"
        results_folder = f"figures/tavi_TechFail/{outcome}"
        os.makedirs(results_folder, exist_ok=True)

        fs_folders = [f.path for f in os.scandir(path) if f.is_dir()]

        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["feature_selection", "model"])
        df_metrics = pd.DataFrame(index=index, columns=['AUC', 'Sen', 'Spe', 'youden_index', 'f1-score', 'ppv',
                                                        'aupr', 'accuracy', 'tp', 'tn', 'fp', 'fn', 'best_test_thresh'])

        # Read excel with validation-based cutoffs
        df_cutoff = pd.read_excel(join(path, "extracted_cv_val_Cutoff.xlsx"))
        df_cutoff["fs"] = df_cutoff["file_path"].apply(lambda x: x.split("\\")[-3].split("_")[0])
        df_cutoff["model"] = df_cutoff["file_path"].apply(lambda x: x.split("\\")[-2])
        # Compute metrics for each model
        roc_list = []
        names_best = []
        predictions_labels = {}
        for fs_folder in fs_folders:
            feature_selection = os.path.basename(fs_folder).split("_")[0]
            # Read train and test datasets for SHAP computation
            train = pd.read_csv(join(fs_folder, f'{feature_selection}_train_feature.csv'), index_col=0)
            test = pd.read_csv(join(fs_folder, f'{feature_selection}_test_feature.csv'), index_col=0)
            # Obtain model folders (remove DT and NB models)
            model_folders = [f.path for f in os.scandir(fs_folder) if f.is_dir()]
            model_folders = [folder for folder in model_folders if "DT" not in folder and "NB" not in folder]
            for model_folder in model_folders:
                model = os.path.basename(model_folder)
                df = pd.read_csv(join(model_folder, "test_prediction.csv"))
                threshold = df_cutoff[(df_cutoff["model"] == model) & (df_cutoff["fs"] == feature_selection)]["Value"].values[0]
                # print feature_selection, model, folder, Sen, specificity
                metrics, y_true, y_pred, label_pred = compute_metrics(y_true=df["Label"], y_pred=df["Pred"], thresh=threshold)
                df_metrics.loc[(feature_selection, model), :] = metrics
                predictions_labels[f"{feature_selection} - {model}"] = y_pred

                # Plot ROC curve for single models
                plt.figure()
                roc = compute_roc(X=y_pred, y=y_true, pos_label=1)
                plot_roc(roc)
                plt.title(f"{feature_selection} - {model}")
                plt.savefig(join(results_folder, f"roc_curve_{feature_selection}_{model}.png"), dpi=300)
                plt.close()
                # Bootstrap and confidence intervals
                plt.figure()
                plot_roc_bootstrap(X=y_pred, y=y_true, pos_label=1, n_bootstrap=1000, show_boots=False, show_ti=False)
                plt.title(f"{feature_selection} - {model}")
                plt.savefig(join(results_folder, f"roc_curve_bootstrap_{feature_selection}_{model}.png"), dpi=300)
                plt.close()

                # Plot confusion matrix
                plot_confusion_matrix(predictions=label_pred, labels=y_true,
                                      output_path=join(results_folder, f"conf_matrix_{feature_selection}_{model}.png"),
                                      title=f"{feature_selection} - {model}", perc=True)  # - Cut-off: {threshold:.3f}

                # Compute and plot SHAP values
                if (outcome == "Vasc" and model == "RF") or (outcome == "Cardiac" and ((model == "RF" and feature_selection in ["KW", "Relief"]) or (model == "LR" and feature_selection == "RFE") or (model == "LDA" and feature_selection == "ANOVA"))):
                # if outcome == "Vasc" and model == "RF" and feature_selection == "RFE":
                    # Save ROC curve for best models
                    roc_list.append(roc)
                    names_best.append(f"{feature_selection} - {model}")
                    # SHAP values
                    print(f"Compute SHAP values for {feature_selection} - {model}")
                    interpreter = Interpretability()
                    output_path = join(results_folder, f"shap_{feature_selection}_{model}.png")
                    with open(join(model_folder, 'model.pickle'), 'rb') as f:
                        model = pickle.load(f)
                    interpreter(output_path, train, test, model, n_train_samples=200)

        # print best roc curve for each feature selection
        plt.figure()
        for i, roc in enumerate(roc_list):
            plot_roc(roc, label=f"{names_best[i]}", color=color_palette[i])
        plt.title(f"{outcome}")
        plt.savefig(join(results_folder, f"roc_curves_best.png"), dpi=300)
        plt.close()

        # Convert to float and resset index
        df_metrics = df_metrics.astype(float).reset_index()
        df_metrics.to_csv(join(results_folder, "metrics.csv"), index=False)

        # Plot heatmap with metrics
        metrics = ["Sen", "Spe", "AUC", "accuracy", "ppv"]
        metric_names = ["Sensitivity", "Specificity", "AUC", "Accuracy", "PPV"]
        colors = ["Reds", "Greens", "Blues", "Purples", "Oranges", "Greys"]
        for i in range(len(metrics)):
            metric = metrics[i]
            color = colors[i]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            df = df_metrics.pivot(values=metric, columns="feature_selection", index="model")
            sns.heatmap(df, cmap=color, annot=True, ax=ax, annot_kws={"fontsize": 14}, cbar=False, fmt='.2g')
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, weight="bold", fontsize=13)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, weight="bold", fontsize=13)
            ax.set_title(f"{metric_names[i]}", weight="bold", fontsize=17)
            plt.savefig(f"{results_folder}/{metric_names[i]}", bbox_inches='tight', dpi=300)

        # Compute DeLong test and plot heatmap
        for best in [False, True]:
            models = names_best if best else list(predictions_labels.keys())
            for binarize in [False, True]:
                df_pvalue = pd.DataFrame(columns=models, index=models)
                df_color = df_pvalue.copy().astype(float)
                for i, model1 in enumerate(models):
                    for model2 in models[i+1:]:
                        y_pred1 = predictions_labels[model1]
                        y_pred2 = predictions_labels[model2]
                        p_value = 10**delong_roc_test(y_true, y_pred1, y_pred2)[0][0]
                        df_pvalue.loc[model2, model1] = np.round(p_value, decimals=2) if p_value >= 0.01 else "<0.01"
                        if binarize:
                            df_color.loc[model2, model1] = 1 if p_value < 0.05 else 0
                        else:
                            df_color.loc[model2, model1] = p_value

                df_pvalue = df_pvalue.iloc[1:, :-1]
                df_color = df_color.iloc[1:, :-1]

                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                fontsize = 18 if best else 5
                label_size = 15 if best else 8
                sns.heatmap(df_color, cmap="Blues", annot=df_pvalue, ax=ax, annot_kws={"fontsize": fontsize}, fmt="", cbar=False)
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.tick_params(axis='both', which='both', length=0)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, weight="bold", fontsize=label_size)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, weight="bold", fontsize=label_size)
                ax.set_title(f"{outcome}", weight="bold", fontsize=17)
                filename = "pvalues_binarized" if binarize else "pvalues"
                filename = filename + "_best" if best else filename
                plt.savefig(f"{results_folder}/{filename}.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import matplotlib


def plot_confusion_matrix(predictions: ArrayLike, labels: ArrayLike, output_path: str, title: str,
                          title_size: float = 16, size: float = 20, label_size: int = 20, dpi: int = 300,
                          perc: bool = False):
    """
    Plot and save a confusion matrix with customizable appearance.
    :param predictions: Predicted labels.
    :param labels: True labels.
    :param output_path: Path to save the confusion matrix plot. If None, the image is not saved.
    :param title: Title of the plot.
    :param title_size: Font size for the title and axis labels (default is 16).
    :param size: Font size for confusion matrix text (default is 20).
    :param label_size: Font size for x/y labels (default is 20).
    :param dpi: Output figure's dpi (default is 300).
    :param perc: If True, plot percentages instead of counts.
    :return:
    """
    matplotlib.rcParams.update({'font.size': size})
    cm = confusion_matrix(y_pred=predictions, y_true=labels, normalize='true') if perc else confusion_matrix(y_pred=predictions, y_true=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.PuBuGn, colorbar=False)
    plt.title(title, fontsize=title_size)
    plt.grid(False)
    plt.tight_layout()
    plt.yticks(fontsize=size)
    plt.xticks(fontsize=size)
    disp.ax_.xaxis.label.set_fontsize(label_size)
    disp.ax_.yaxis.label.set_fontsize(label_size)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    matplotlib.rcdefaults()

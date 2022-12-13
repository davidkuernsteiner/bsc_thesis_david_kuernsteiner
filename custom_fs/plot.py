import matplotlib.pyplot as plt
import matplotlib as mpl
from fs_mol.data import FSMolDataset, DataFold


def plot_dauprc_boxplot(dauprc, model_names, DataFold, n_support):

    if DataFold == DataFold.VALIDATION:
        fold = "Validation"
    else:
        fold = "Test"

    fig, ax = plt.subplots()
    ax.set_title(f"DeltaAUPRC on {fold} set | support set size = {n_support}")
    ax.boxplot(dauprc, vert=False, labels=model_names)
    ax.set_xlabel("DeltaAUPRC")

    return fig


def plot_avg_dauprc(dauprc, model_names, DataFold, n_support):

    if DataFold == DataFold.VALIDATION:
        fold = "Validation"
    else:
        fold = "Test"

    symbols = ["o", "s", "^"]
    fig, ax = plt.subplots()
    ax.set_title(f"Average DeltaAUPRC on {fold} set")
    for i, model in enumerate(model_names):
        ax.plot(n_support, dauprc[i], label=model, marker=symbols[i])

    plt.xticks(n_support)
    ax.grid()
    ax.set_xlabel("support set size")
    ax.set_ylabel("DeltaAUPRC")

    ax.legend(loc="lower right", prop={"size": 12})

    return fig


"""d = [[0.1, 0.15, 0.17, 0.2, 0.22], [0.15, 0.25, 0.3, 0.32, 0.35]]
n = [16, 32, 64, 128, 256]
m = ["MT", "MAML"]

fig = plot_avg_dauprc(d, m, DataFold.TEST, n)
fig.show()"""

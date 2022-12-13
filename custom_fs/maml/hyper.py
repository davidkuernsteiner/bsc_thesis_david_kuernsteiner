import os
import optuna
import torch
import wandb
import json
from custom_fs.maml.data import MAMLDataset
from custom_fs.maml.train import *
from custom_fs.plot import *


def maml_hyperparameter_search(dataset: MAMLDataset, seed):

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study_name = "MAML"
    objective = MAMLTrainLoop(dataset, study_name=study_name, max_epochs=10, n_support=16, seed=seed)

    study = optuna.create_study(sampler=sampler,
                                pruner=pruner,
                                study_name=study_name,
                                direction="maximize")

    study.optimize(objective, n_trials=100, callbacks=[objective.checkpoint_callback], gc_after_trial=True)

    config = dict(study.best_trial.params)
    with open(f"{study_name}_best_hyperparameters.json", "w") as outfile:
        json.dump(config, outfile)

    final_test_run = wandb.init(project="Metalearning for drug discovery",
                                entity="davidkuernsteiner",
                                config=config,
                                group=study_name,
                                name=f"{study_name}_final_test",
                                reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(os.getcwd(), "best_models", f"{study_name}"))
    model = model.to(device)
    objective.model = model

    for n_supp in [16, 32, 64, 128, 256]:

        summary = {"avg_dauprc": [[]],
                   "dauprc_list": []}
        total_dauprc = []
        i = 0

        objective.dataset.build_finetuning_set(n_supp, DataFold.TEST)
        for finetune_train_loader, finetune_test_loader, stats, task_name in tqdm(objective.dataset.finetuning_set,
                                                                                  desc="Test progress"):
            dauprc = objective.finetune_on_task(finetune_train_loader, finetune_test_loader, stats)
            total_dauprc.append(dauprc)
            i += 1
        avg_dauprc = sum(total_dauprc) / i
        summary["avg_dauprc"].append(avg_dauprc)
        summary["dauprc_list"].append(total_dauprc)

    boxplot = plot_dauprc_boxplot(summary["dauprc_list"][0], ["MAML"], DataFold.TEST, 16)
    avg_plot = plot_avg_dauprc(summary["avg_dauprc"], ["MAML"], DataFold.TEST, [16, 32, 64, 128, 256])
    wandb.log({"DeltaAUPRC boxplot": wandb.Image(boxplot),
               "Average DaltaAUPRC": wandb.Image(avg_plot),
               "Hyperparameter Importance": wandb.Image(optuna.visualization.plot_param_importances(study))})

    plt.close(boxplot)
    plt.close(avg_plot)

    final_test_run.finish()

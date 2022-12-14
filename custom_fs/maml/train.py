import os
import torch.optim
from custom_fs.models import HyperFCDNN
from custom_fs.metrics import DeltaAUPRC
from custom_fs.maml.data import *
from tqdm import tqdm
import wandb
import optuna
from custom_fs.plot import *
import copy
from learn2learn.algorithms import MAML


class MAMLTrainLoop:

    def __init__(self, dataset: MAMLDataset, study_name, max_epochs, n_support=16, seed=123):

        self.dataset = dataset
        self.max_epochs = max_epochs
        self.n_support = n_support
        self.seed = seed
        self.study_name = study_name

    def __call__(self, trial):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Using device: {device}")
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.metric = DeltaAUPRC()

        model = HyperFCDNN(trial).to(device)
        maml = MAML(model, lr=trial.suggest_float("meta_lr", 0.0001, 0.001))
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 0.0005, 0.005))
        task_batch_size = trial.suggest_int("task_batch_size", 1, 10)
        k = trial.suggest_int("k", 2, 10)
        adaptation_steps = trial.suggest_int("adaptation_steps", 1, 10)

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(project="Metalearning for drug discovery",
                   entity="davidkuernsteiner",
                   config=config,
                   group=self.study_name,
                   name=f"{self.study_name}_run_{trial.number}",
                   reinit=True)

        best_avg_dauprc = -np.inf

        for epoch in range(self.max_epochs):

            train_loss = 0
            self.dataset.build_train_set(k)
            for iteration in tqdm(range(self.dataset.n_train_tasks // task_batch_size), desc="Train progress"):

                optimizer.zero_grad()

                for task in range(task_batch_size):
                    task = next(self.dataset.train_set)
                    learner = maml.clone()
                    eval_error = self.adapt_to_task(task, learner, adaptation_steps=adaptation_steps)
                    train_loss += eval_error
                    eval_error.backward()

                for p in maml.parameters():
                    p.grad.data.mul_(1.0 / task_batch_size)
                optimizer.step()

            avg_train_loss = train_loss / (self.dataset.n_train_tasks // task_batch_size)

            print(f"epoch {epoch + 1}: average train loss {avg_train_loss}")

            total_dauprc = []
            i = 0

            self.dataset.build_finetuning_set(self.n_support, DataFold.VALIDATION)

            for finetune_train_loader, finetune_test_loader, stats, task_name in tqdm(self.dataset.finetuning_set,
                                                                                      desc="Validation progress"):
                self.model = copy.deepcopy(model)
                dauprc = self.finetune_on_task(finetune_train_loader, finetune_test_loader, stats)
                total_dauprc.append(dauprc)
                i += 1

            avg_dauprc = sum(total_dauprc) / i
            print(f"Performance on Validation Set:\n"
                  f"Average DeltaAUPRC: {avg_dauprc}\n"
                  f"Min DeltaAUPRC: {min(total_dauprc)}\n"
                  f"Max DeltaAUPRC: {max(total_dauprc)}")

            if avg_dauprc > best_avg_dauprc:
                self.best_model = copy.deepcopy(model)
                best_avg_dauprc = avg_dauprc

            if trial:
                trial.report(avg_dauprc, epoch)

                wandb.log(data={"average train loss": avg_train_loss, "average deltaAUPRC": avg_dauprc},
                          step=epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return best_avg_dauprc

    def adapt_to_task(self, data, learner, adaptation_steps):

        x_supp, y_supp, x_query, y_query = tuple(array.to(self.device) for array in data[0])

        for step in range(adaptation_steps):
            pred = torch.reshape(learner(x_supp), y_supp.shape)
            train_error = self.loss_fn(pred, y_supp)
            learner.adapt(train_error)

        pred = torch.reshape(learner(x_query), y_query.shape)
        valid_error = self.loss_fn(pred, y_query)

        return valid_error

    def finetune_on_task(self, _train_loader, _test_loader, pos_label_ratio):

        finetune_loss_fn = torch.nn.BCEWithLogitsLoss()
        metric = DeltaAUPRC()

        finetune_optimizer = torch.optim.Adam(self.model.parameters())

        for i in range(10):

            for x_train, y_train in _train_loader:
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                pred = torch.reshape(self.model(x_train), y_train.shape)
                loss = finetune_loss_fn(pred, y_train)
                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()

        self.model.eval()
        for x_test, y_test in _test_loader:
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            with torch.no_grad():
                pred = torch.sigmoid(torch.reshape(self.model(x_test), y_test.shape))
                metric.update(pred, y_test)
        self.model.train()

        _dauprc = metric.compute(pos_label_ratio["test"])

        return _dauprc.data.cpu().numpy()

    def checkpoint_callback(self, study, trial):

        if study.best_trial == trial:
            torch.save(self.best_model, os.path.join(os.getcwd(), "best_models", f"{self.study_name}"))

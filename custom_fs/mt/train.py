import os
import torch.optim
from torch.utils.data import DataLoader
from custom_fs.models import MTWrapper, HyperFCDNN
from custom_fs.metrics import DeltaAUPRC, MaskedBCE
from custom_fs.mt.data import *
from tqdm import tqdm
import wandb
import optuna
from custom_fs.plot import *
import copy


class MTTrainLoopSigmoid:

    def __init__(self, dataset: MTDataset, study_name, max_epochs, n_support=256, seed=123):

        self.dataset = dataset
        self.max_epochs = max_epochs
        self.n_support = n_support
        self.seed = seed
        self.study_name = study_name

    def __call__(self, trial):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Using device: {device}")
        train_loss_fn = MaskedBCE(reduction="sum")

        model = MTWrapper(4938, trial).to(device)
        train_optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 0.0005, 0.005))
        train_loader = DataLoader(self.dataset.train_set, batch_size=trial.suggest_int("batch_size", 256, 4096),
                                  shuffle=True,
                                  num_workers=1, pin_memory=False)

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(project="Metalearning for drug discovery",
                   entity="davidkuernsteiner",
                   config=config,
                   group=self.study_name,
                   name=f"{self.study_name}_run_{trial.number}",
                   reinit=True)

        model.device = device
        self.best_model = copy.deepcopy(model)
        best_avg_dauprc = -np.inf

        def train_step(x, labels, ids):

            y = get_label_matrix(labels, ids, 4938)
            loss_mask = get_loss_mask(y)
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            pred = model(x)
            loss = train_loss_fn(pred, y, weight=loss_mask)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            return loss

        for epoch in range(1, self.max_epochs):

            model.pretrain()
            total_train_loss = 0
            for features, labels, task_ids in tqdm(train_loader, desc=f"Epoch {epoch} progress", total=len(train_loader)):

                batch_loss = train_step(features, labels, task_ids)
                total_train_loss += batch_loss

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch} average train loss: {avg_train_loss}")

            self.model = model
            total_dauprc = []
            i = 0
            self.dataset.build_finetuning_set(self.n_support, DataFold.VALIDATION)

            for finetune_train_loader, finetune_test_loader, stats, task_name in tqdm(self.dataset.finetuning_set,
                                                                                      desc="Validation progress"):

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
            gc.collect()

        return best_avg_dauprc

    def finetune_on_task(self, _train_loader, _test_loader, pos_label_ratio):

        finetune_loss_fn = torch.nn.BCEWithLogitsLoss()
        metric = DeltaAUPRC()

        self.model.finetune()
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


class MTTrainLoopSoftmax:

    def __init__(self, dataset: MTDataset, study_name, max_epochs, n_support=256, seed=123):

        self.dataset = dataset
        self.max_epochs = max_epochs
        self.n_support = n_support
        self.seed = seed
        self.study_name = study_name

    def __call__(self, trial):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Using device: {device}")
        train_loss_fn = torch.nn.CrossEntropyLoss()

        model = MTWrapper(4938, trial).to(device)
        train_optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 0.0005, 0.005))
        train_loader = DataLoader(self.dataset.train_set, batch_size=trial.suggest_int("batch_size", 256, 4096),
                                  shuffle=True,
                                  num_workers=1, pin_memory=False)

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(project="Metalearning for drug discovery",
                   entity="davidkuernsteiner",
                   config=config,
                   group=self.study_name,
                   name=f"{self.study_name}_run_{trial.number}",
                   reinit=True)

        model.device = device
        self.best_model = copy.deepcopy(model)
        best_avg_dauprc = -np.inf

        def train_step(x, ids):

            x, y = x.to(device), ids.to(device)
            pred = model(x)
            loss = train_loss_fn(pred, y)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            return loss

        for epoch in range(1, self.max_epochs):

            model.pretrain()
            total_train_loss = 0
            for features, labels, task_ids in tqdm(train_loader, desc=f"Epoch {epoch} progress", total=len(train_loader)):

                batch_loss = train_step(features, task_ids)
                total_train_loss += batch_loss

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch} average train loss: {avg_train_loss}")

            self.model = model
            total_dauprc = []
            i = 0
            self.dataset.build_finetuning_set(self.n_support, DataFold.VALIDATION)

            for finetune_train_loader, finetune_test_loader, stats, task_name in tqdm(self.dataset.finetuning_set,
                                                                                      desc="Validation progress"):

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
            gc.collect()

        return best_avg_dauprc

    def finetune_on_task(self, _train_loader, _test_loader, pos_label_ratio):

        finetune_loss_fn = torch.nn.BCEWithLogitsLoss()
        metric = DeltaAUPRC()

        self.model.finetune()
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


from torch import nn
from optuna import trial


class HyperFCDNN(nn.Module):

    def __init__(self, trial: trial):
        super(HyperFCDNN, self).__init__()

        n_layers = trial.suggest_int("n_layers", 1, 2)
        layers = []
        in_features = 2248

        p = trial.suggest_float("dropout_in", 0.1, 0.2)
        layers.append(nn.Dropout(p))
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 500, 3000)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features

        self.hidden_stack = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.hidden_stack(x)
        x = self.head(x)
        return x


class FCDNN(nn.Module):

    def __init__(self):
        super(FCDNN, self).__init__()

        self.hidden_stack = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(2248, 1024), nn.ReLU(),
            nn.Dropout(p=0.5), nn.Linear(1024, 1024), nn.ReLU())

        self.head = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.hidden_stack(x)
        x = self.head(x)
        return x


class MTWrapper(HyperFCDNN):

    def __init__(self, n_tasks, trial):
        super(MTWrapper, self).__init__(trial)
        self.device = None
        self.n_tasks = n_tasks
        self.train_head = nn.Linear(self.head.in_features, n_tasks)
        self.head = None

    def pretrain(self):

        self.head = self.train_head
        self.head.to(self.device)

        for param in self.parameters():
            param.requires_grad = True

    def finetune(self):

        for param in self.parameters():
            param.requires_grad = False

        self.head = nn.Linear(self.head.in_features, 1)
        self.head.to(self.device)


class MAMLWrapper():

    pass

import copy
import os.path

from custom_fs.models import MTWrapper
from custom_fs.plot import *
from custom_fs.metrics import DeltaAUPRC
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from mergedeep import merge, Strategy
from itertools import chain


class EvalModel:

    def __init__(self, model, metric, dataset):

        self.model = model
        self.task_model = None
        self.metric = metric
        self.key_metric = DeltaAUPRC()
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_by_finetuning_on_task(self, support_set, query_set, pos_label_ratio):

        finetune_loss_fn = torch.nn.BCEWithLogitsLoss()
        metric = self.metric.to(self.device)
        key_metric = self.key_metric

        finetune_optimizer = torch.optim.Adam(self.task_model.parameters())

        for i in range(10):

            for x_supp, y_supp in support_set:
                x_supp, y_supp = x_supp.to(self.device), y_supp.to(self.device)
                pred = torch.reshape(self.task_model(x_supp), y_supp.shape)
                loss = finetune_loss_fn(pred, y_supp)
                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()

        self.task_model.eval()
        key_metric.reset()
        metric.reset()
        for x_query, y_query in query_set:
            x_query, y_query = x_query.to(self.device), y_query.to(self.device)
            with torch.no_grad():
                pred = torch.sigmoid(torch.reshape(self.task_model(x_query), y_query.shape))
                key_metric.update(pred, y_query)
                metric.update(pred, y_query)
        self.task_model.train()

        _key_metric = key_metric.compute(pos_label_ratio["test"])

        _metric = {**metric.compute(), **{"DeltaAUPRC": _key_metric}}
        _metric = {key: float(value.data.cpu()) for key, value in _metric.items()}

        return _metric

    def eval_model(self, model_name):

        summary = []
        for n_supp in [16, 32, 64, 128, 256]:

            total_metrics = []
            i = 0

            self.dataset.build_finetuning_set(n_supp, DataFold.TEST)
            for finetune_train_loader, finetune_test_loader, stats, task_name in tqdm(self.dataset.finetuning_set,
                                                                                      desc="Test progress"):
                if isinstance(self.model, MTWrapper):
                    self.task_model = self.model
                    self.task_model.finetune()
                else:
                    self.task_model = copy.deepcopy(self.model)
                metrics = self.eval_by_finetuning_on_task(finetune_train_loader, finetune_test_loader, stats)
                total_metrics.append(metrics)
                dd = defaultdict(list)
                i += 1

            for d in total_metrics:
                for key, value in d.items():
                    dd[key].append([value])

            nsupp_summary = {"avg_metrics": {key: [sum(list(chain(*value))) / i] for key, value in dd.items()},
                             "metrics_list": dd}

            print(nsupp_summary)

            summary.append(nsupp_summary)

        summary = merge({}, *summary, strategy=Strategy.ADDITIVE)

        with open(os.path.join("model_metrics", f"{model_name}_best_model_metrics.json"), "w") as outfile:
            json.dump(summary, outfile)

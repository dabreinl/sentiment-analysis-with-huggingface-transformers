import torch
import mlflow
import pandas as pd
from torch import nn
from datasets import load_metric
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score


def compute_metrics(
    predictions, batch, num_classes=6, task="multiclass"
):  # TODO remove hardcoding of classes maybe just implement sklearn versions
    predictions = predictions.cpu()
    true_label = batch["labels"].cpu()

    f1 = F1Score(task=task, num_classes=num_classes, average="weighted")
    accuracy = Accuracy(task=task, num_classes=num_classes)

    f1_score = f1(predictions, true_label).item()
    acc_score = accuracy(predictions, true_label).item()

    return {"f1": f1_score, "acc": acc_score}


class Model_training:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_phase(self, optimizer, train_dataloader, lr_scheduler=False):
        self.model = self.model.to(self.device)

        train_loss = 0
        f1_score = 0
        acc_score = 0

        for batch in tqdm(train_dataloader, leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.model.train()

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            predictions = torch.argmax(outputs["logits"], dim=1)

            loss = outputs.loss

            train_loss += loss.item()

            metrics = compute_metrics(batch=batch, predictions=predictions)
            f1_score += metrics["f1"]
            acc_score += metrics["acc"]

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if lr_scheduler:
                lr_scheduler.step()

        train_loss = train_loss / len(train_dataloader)
        f1_score = f1_score / len(train_dataloader)
        acc_score = acc_score / len(train_dataloader)

        return train_loss, f1_score, acc_score

    def eval_phase(self, eval_dataloader):
        eval_loss = 0
        f1_score = 0
        acc_score = 0

        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

            loss = outputs.loss

            eval_loss += loss.item()

            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=1)

            metrics = compute_metrics(batch=batch, predictions=predictions)
            f1_score += metrics["f1"]
            acc_score += metrics["acc"]

        eval_loss = eval_loss / len(eval_dataloader)
        f1_score = f1_score / len(eval_dataloader)
        acc_score = acc_score / len(eval_dataloader)

        return eval_loss, f1_score, acc_score

    def train(
        self,
        epochs,
        train_dataloader,
        eval_dataloader,
        optimizer,
        early_stopper=None,
        model_save_name=None,
        scheduler=False,
    ):
        results = {
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
            "lrs": [],
        }

        if model_save_name is None:
            model_name = "unkown"
        else:
            model_name = model_save_name

        with mlflow.start_run(
            run_name=f"{model_name}-{pd.to_datetime('today').isoformat()}"
        ):
            run = mlflow.active_run()
            run_id = run.info.run_id
            print(f"\nrun_id: {run_id}; status{run.info.status}")

            for epoch in range(epochs):
                train_loss, train_f1, train_acc = self.train_phase(
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    lr_scheduler=scheduler,
                )
                eval_loss, eval_f1, eval_acc = self.eval_phase(
                    eval_dataloader=eval_dataloader
                )
                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"\nEpoch {epoch+1}, Current LR: {current_lr}:\nTrain Loss: {train_loss:.5f} | Train F1: {train_f1:.5f} | Train Acc: {train_acc:.5f}\nValidation Loss: {eval_loss:.5f} | Validation F1: {eval_f1:.5f} | Validation Acc: {eval_acc:.5f}\n"
                )

                # Update results dictionary
                # TODO maybe just use zip() and iterate
                results["train_loss"].append(train_loss)
                results["train_f1"].append(train_f1)
                results["val_loss"].append(eval_loss)
                results["val_f1"].append(eval_f1)
                results["lrs"].append(current_lr)

                if early_stopper is not None:
                    if early_stopper.early_stop(eval_loss):
                        print("Stopping Training..")
                        break

                mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                mlflow.log_metric(key="train_f1", value=train_f1, step=epoch)
                mlflow.log_metric(key="val_loss", value=eval_loss, step=epoch)
                mlflow.log_metric(key="val_f1", value=eval_f1, step=epoch)

                if model_save_name is not None:
                    path_to_model = f"app/modeling/models/{model_save_name}.pth"
                    torch.save(
                        self.model.state_dict(),
                        path_to_model,
                    )

        return results

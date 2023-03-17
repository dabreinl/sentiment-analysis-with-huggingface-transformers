import torch
from torch import nn
from datasets import load_metric
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(predictions, batch):
    f1 = load_metric("f1")
    accuracy = load_metric("accuracy")

    f1.add_batch(predictions=predictions, references=batch["sentiment"])
    accuracy.add_batch(predictions=predictions, references=batch["sentiment"])

    return {"f1": f1.compute(), "acc": accuracy.compute()}


class Model_training:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_phase(self, optimizer, train_dataloader, lr_scheduler=None):
        self.model = self.model.to(self.device)

        train_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.model.train()

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["sentiment"],
            )

            loss = outputs.loss

            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        train_loss = train_loss / len(train_dataloader)

        return train_loss

    def eval_phase(self, eval_dataloader):
        eval_loss = 0
        # f1_score = 0
        # acc_score = 0

        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["sentiment"],
                )

            loss = outputs.loss

            eval_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits)

            # TODO: see how we can fix
            # metrics = compute_metrics(predictions, batch)
            # f1_score += metrics.f1
            # acc_score += metrics.acc

        # avg_metrics = {"f1": f1_score / len(eval_dataloader),
        #               "acc": acc_score / len(eval_dataloader),
        #               "loss": eval_loss / len(eval_dataloader)}
        eval_loss = eval_loss / len(eval_dataloader)

        return eval_loss

    def train(self, epochs, train_dataloader, eval_dataloader, optimizer):
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_phase(
                optimizer=optimizer, train_dataloader=train_dataloader
            )
            eval_loss = self.eval_phase(eval_dataloader=eval_dataloader)

            print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Loss: {eval_loss}")

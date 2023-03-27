import torch
import transformers
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel, DataCollatorWithPadding
from torch.utils.data import DataLoader
from modeling.model import TweetClassificationModel
from modeling.train import Model_training


class Training:
    def __init__(self, model_checkpoint, early_stopper=None):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.early_stopper = early_stopper

    def dataloader(self, dataset_name: str):
        self.ds = load_dataset(dataset_name)
        self.class_names = self.ds["train"].features["label"].names

        return None

    def tokenize_batch(self, batch):
        return self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def create_encoded_ds(self):
        self.ds_encoded = self.ds.map(
            self.tokenize_batch, batched=True, batch_size=None
        )

        self.ds_encoded.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

        return None

    def create_dataloader(self, batch_size, encoded_data):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataloader = DataLoader(
            encoded_data["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        self.val_dataloader = DataLoader(
            encoded_data["validation"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        self.test_dataloader = DataLoader(
            encoded_data["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        return None

    def load_model(self):
        model = AutoModel.from_pretrained(self.model_checkpoint)
        model_config = AutoConfig.from_pretrained(self.model_checkpoint)

        custom_model = TweetClassificationModel(
            checkpoint=self.model_checkpoint, num_classes=len(self.class_names)
        )

        self.model = custom_model

        return None

    def train_model(self, lr=1e-5, epochs=5, device=torch.device("mps")):
        parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(parameters, lr)

        trainer = Model_training(model=self.model, device=device)

        train_results = trainer.train(
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.val_dataloader,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=self.early_stopper,
            model_save_name="test-can-be-deleted",
        )

        return train_results


if __name__ == "__main__":
    print("\ncreating training object..")
    training = Training("distilbert-base-uncased")
    print("\nloading data..")
    training.dataloader("emotion")
    print("\nencoding tweets..")
    training.create_encoded_ds()
    print("\ncreating dataloader..")
    training.create_dataloader(32, training.ds_encoded)
    print("\nloading model..")
    training.load_model()
    print("\ntraining model..")
    results = training.train_model()

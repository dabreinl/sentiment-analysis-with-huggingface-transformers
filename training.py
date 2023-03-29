import torch
import transformers
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel, DataCollatorWithPadding
from torch.utils.data import DataLoader
from app.modeling.model import TweetClassificationModel
from app.modeling.train import Model_training
from app.modeling.model_utils import EarlyStopper


class Training:
    def __init__(
        self, model_checkpoint, early_stopper=None, device=torch.device("mps")
    ):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.early_stopper = early_stopper
        self.device = device

    def _dataloader(self, dataset_name: str):
        print("\nloading data..")
        self.ds = load_dataset(dataset_name)
        self.class_names = self.ds["train"].features["label"].names

    def _tokenize_batch(self, batch):
        return self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def _create_encoded_ds(self):
        print("\nencoding tweets..")
        self.ds_encoded = self.ds.map(
            self._tokenize_batch, batched=True, batch_size=None
        )

        self.ds_encoded.set_format(
            "torch", columns=["input_ids", "attention_mask", "label"]
        )

    def _create_dataloader(self, batch_size, encoded_data):
        print("\ncreating dataloader..")
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

    def _load_model(self, load_model=True, saved_model_name=None):
        print("\nloading model..")
        model = AutoModel.from_pretrained(self.model_checkpoint)
        custom_model = TweetClassificationModel(
            checkpoint=self.model_checkpoint, num_classes=len(self.class_names)
        )

        if load_model:
            custom_model.load_state_dict(
                torch.load(f"app/modeling/models/{saved_model_name}.pth")
            )

        self.model = custom_model

    def _train_model(self, model_name, epochs, lr=1e-5):
        print("\ntraining model..")
        parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(parameters, lr)

        trainer = Model_training(model=self.model, device=self.device)

        train_results = trainer.train(
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.val_dataloader,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=self.early_stopper,
            model_save_name=model_name,
        )


if __name__ == "__main__":
    early_stopper = EarlyStopper(patience=0)
    training = Training("distilbert-base-uncased", early_stopper=early_stopper)
    training._dataloader("emotion")
    training._create_encoded_ds()
    training._create_dataloader(32, training.ds_encoded)
    training._load_model(
        load_model=True,
        saved_model_name="distilbert-base-finetuned-for-tweet-classification",
    )
    results = training._train_model("can-be-deleted", epochs=1)

import torch
import transformers
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from training_config import TrainingConfig
from app.modeling.model import TweetClassificationModel
from app.modeling.train import Model_training
from app.modeling.utils.model_utils import EarlyStopper
from app.modeling.utils.data_utils import (
    get_balanced_dataset_random_oversampler,
    create_balanced_datasets,
)


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

    def _create_encoded_ds(self, imbalanced=False, balancer="random_oversampler"):
        print("\nencoding tweets..")
        if imbalanced:
            if balancer == "random_oversampler":
                self.ds_encoded = get_balanced_dataset_random_oversampler(
                    self.ds, self.tokenizer
                )
            elif balancer == "augmentation":
                self.ds = create_balanced_datasets(self.ds)
                self.ds.set_format("pandas")
                print(self.ds["train"][:][["label"]].value_counts())
                self.ds.reset_format()
                self.ds_encoded = self.ds.map(
                    self._tokenize_batch, batched=True, batch_size=None
                )  # TODO repitition
                self.ds_encoded.set_format(
                    "torch", columns=["input_ids", "attention_mask", "label"]
                )
            else:
                raise ValueError(
                    "balancer must be one of the following: ['random_oversampler', 'augmentation']"
                )
        else:
            self.ds_encoded = self.ds.map(
                self._tokenize_batch, batched=True, batch_size=None
            )  # TODO: Could just put this before if -> is double

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

    def _train_model(self, model_name, epochs, lr=1e-5, scheduler=False):
        print("\ntraining model..")
        parameters = self.model.parameters()

        optimizer = torch.optim.AdamW(parameters, lr)

        # If we use scheduler set the warmup steps to 10% of total training steps
        if scheduler:
            num_training_steps = len(self.train_dataloader) * epochs
            num_warmup_steps = int(num_training_steps * 0.1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        trainer = Model_training(model=self.model, device=self.device)

        train_results = trainer.train(
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.val_dataloader,
            optimizer=optimizer,
            epochs=epochs,
            early_stopper=self.early_stopper,
            model_save_name=model_name,
            scheduler=scheduler,
        )


if __name__ == "__main__":
    config = TrainingConfig()
    early_stopper = EarlyStopper(patience=config.early_stopping_patience)
    training = Training(config.model_checkpoint, early_stopper=early_stopper)
    training._dataloader(config.dataset_name)
    training._create_encoded_ds(imbalanced=config.imbalanced, balancer=config.balancer)
    training._create_dataloader(config.batch_size, training.ds_encoded)
    training._load_model(
        load_model=config.load_model,
        saved_model_name=config.saved_model_name,
    )
    results = training._train_model(
        config.model_save_name, epochs=config.epochs, scheduler=config.scheduler
    )

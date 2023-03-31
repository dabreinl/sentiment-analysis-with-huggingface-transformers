import torch
import transformers
import pandas as pd
from datasets import load_dataset, DatasetDict
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
        self,
        model_checkpoint: str,
        early_stopper: EarlyStopper = None,
        device: torch.device = torch.device("mps"),
    ):
        """
        Initialize the Training class.

        Args:
            model_checkpoint (str): The model checkpoint to be used for training.
            early_stopper (EarlyStopper, optional): An instance of the EarlyStopper class, used for early stopping during training. Defaults to None.
            device (torch.device, optional): The device to be used for model training. Defaults to torch.device("mps").
        """
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.early_stopper = early_stopper
        self.device = device

    def _load_data(self, dataset_name: str):
        """
        Load the dataset using the given dataset name.

        Args:
            dataset_name (str): The name of the dataset to be loaded.
        """
        print("\nloading data..")
        self.ds = load_dataset(dataset_name)
        self.class_names = self.ds["train"].features["label"].names

    def _tokenize_batch(self, batch: DatasetDict):
        """
        Tokenize a batch of text.

        Args:
            batch (DatasetDict): A DatasetDict containing a batch of text to be tokenized.

        Returns:
            DatasetDict: A dictionary containing the tokenized batch.
        """
        return self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )  # TODO maybe put this into another script as it is used eg by data_utils in random oversampling and also here -> so it does not occur multiple times

    def _create_encoded_ds(
        self, imbalanced: bool = False, balancer: str = "random_oversampler"
    ):
        """
        Create an encoded dataset based on the given parameters.

        Args:
            imbalanced (bool, optional): Whether the dataset is imbalanced or not. Defaults to False.
            balancer (str, optional): The balancing technique to be used if the dataset is imbalanced (random_oversampler or augmentation). Defaults to "random_oversampler".
        """
        print("\nencoding tweets..")
        if imbalanced:
            if balancer == "random_oversampler":
                self.ds_encoded = get_balanced_dataset_random_oversampler(
                    self.ds, self.tokenizer
                )
            elif balancer == "augmentation":
                self.ds = create_balanced_datasets(self.ds)
                self.ds.set_format("pandas")
                print(
                    f'train examples per label:\n{self.ds["train"][:][["label"]].value_counts()}'
                )
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

    def _create_dataloader(self, batch_size: int, encoded_data: DatasetDict):
        """
        Create dataloaders for training, validation, and testing.

        Args:
            batch_size (int): The batch size to be used in the DataLoader.
            encoded_data (DatasetDict): The encoded dataset to be used for creating the dataloaders.
        """
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

    def _load_model(self, load_model: bool = True, saved_model_name: str = None):
        """
        Load the model for training.

        Args:
            load_model (bool, optional): Whether to load a pre-trained model or not. Defaults to True.
            saved_model_name (str, optional): The name of the saved model to be loaded. Defaults to None.
        """
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

    def _train_model(
        self, model_name: str, epochs: int, lr: float = 1e-5, scheduler=False
    ):
        """
        Train the model using the specified parameters.

        Args:
            model_name (str): The name of the model to be saved.
            epochs (int): The number of epochs for training.
            lr (float, optional): The learning rate to be used during training. Defaults to 1e-5.
            scheduler (bool, optional): Whether to use a learning rate scheduler or not. Defaults to False.
        """
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
    training = Training(
        config.model_checkpoint, early_stopper=early_stopper, device=config.device
    )
    training._load_data(config.dataset_name)
    training._create_encoded_ds(imbalanced=config.imbalanced, balancer=config.balancer)
    training._create_dataloader(config.batch_size, training.ds_encoded)
    training._load_model(
        load_model=config.load_model,
        saved_model_name=config.saved_model_name,
    )
    results = training._train_model(
        config.model_save_name, epochs=config.epochs, scheduler=config.scheduler
    )

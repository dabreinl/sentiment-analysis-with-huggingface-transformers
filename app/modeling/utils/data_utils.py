import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler


def get_balanced_dataset(
    dataset, tokenizer, resampler=RandomOverSampler(random_state=42)
):
    # Tokenize the dataset and add 'input_ids' and 'attention_mask' features
    def tokenize_and_encode(batch):
        encoding = tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=128
        )
        return {**batch, **encoding}

    dataset = dataset.map(tokenize_and_encode, batched=True)

    # Convert the HuggingFace Dataset to a list of dictionaries
    def to_list_of_dicts(ds):
        data = ds[:]
        return [dict(zip(data.keys(), row)) for row in zip(*data.values())]

    train_data = to_list_of_dicts(dataset["train"])
    valid_data = to_list_of_dicts(dataset["validation"])
    test_data = to_list_of_dicts(dataset["test"])

    def resample_data(data, labels):
        resampled_data, _ = resampler.fit_resample(
            np.array(data).reshape(-1, 1), labels
        )
        return resampled_data.ravel().tolist()

    train_resampled = resample_data(train_data, dataset["train"]["label"])
    valid_resampled = resample_data(valid_data, dataset["validation"]["label"])
    test_resampled = resample_data(test_data, dataset["test"]["label"])

    # Convert the resampled lists back to HuggingFace DatasetDict
    def from_list_of_dicts(data_list):
        return Dataset.from_dict(
            {k: [d[k] for d in data_list] for k in data_list[0].keys()}
        )

    balanced_dataset = DatasetDict(
        {
            "train": from_list_of_dicts(train_resampled),
            "validation": from_list_of_dicts(valid_resampled),
            "test": from_list_of_dicts(test_resampled),
        }
    )

    balanced_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    return balanced_dataset

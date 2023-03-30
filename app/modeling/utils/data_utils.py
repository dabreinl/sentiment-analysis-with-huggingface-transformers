import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler
from datasets import DatasetDict
from sklearn.utils import shuffle
import pandas as pd


def get_balanced_dataset_random_oversampler(
    dataset, tokenizer, resampler=RandomOverSampler(random_state=42)
):
    # Tokenize the dataset and add 'input_ids' and 'attention_mask' features
    def tokenize_and_encode(batch):
        encoding = tokenizer(
            batch["text"], truncation=True, padding=True, add_special_tokens=True
        )  # TODO think about if it maybe makes sense to remove the max_length and add special add_special_tokens
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

    # Convert the resampled lists back to HuggingFace DatasetDict
    def from_list_of_dicts(data_list):
        return Dataset.from_dict(
            {k: [d[k] for d in data_list] for k in data_list[0].keys()}
        )

    balanced_dataset = DatasetDict(
        {
            "train": from_list_of_dicts(train_resampled),
            "validation": from_list_of_dicts(valid_data),
            "test": from_list_of_dicts(test_data),
        }
    )

    # Set the dataset format to be used by the transformers library
    balanced_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    return balanced_dataset


def balance_dataset(df):
    max_class_size = df["label"].value_counts().max()
    balanced_data = pd.DataFrame()

    print("\naugmenting data..")
    for class_label in df["label"].unique():
        class_data = df[df["label"] == class_label]
        samples_needed = max_class_size - len(class_data)
        augmented_data = augment_data(class_data, samples_needed)
        balanced_class_data = pd.concat([class_data, augmented_data])
        balanced_data = pd.concat([balanced_data, balanced_class_data])

    balanced_data = shuffle(balanced_data)

    return balanced_data


def augment_data(df, num_samples):
    aug = naw.SynonymAug(aug_src="wordnet", aug_p=0.4)
    augmented_data = pd.DataFrame(columns=["text", "label"])

    while len(augmented_data) < num_samples:
        sample = df.sample()
        augmented_text = aug.augment(sample["text"].values[0])
        new_row = pd.DataFrame(
            {"text": [augmented_text], "label": [sample["label"].values[0]]}
        )
        augmented_data = pd.concat([augmented_data, new_row], ignore_index=True)

    return augmented_data


def create_balanced_datasets(dataset_dict):
    train_df = dataset_dict["train"].to_pandas()
    balanced_train_df = balance_dataset(train_df)

    # Convert the "text" column to strings
    balanced_train_df["text"] = balanced_train_df["text"].astype(str)

    balanced_train_hf_dataset = Dataset.from_pandas(balanced_train_df)

    balanced_datasets = DatasetDict(
        {
            "train": balanced_train_hf_dataset,
            "validation": dataset_dict["validation"],
            "test": dataset_dict["test"],
        }
    )

    return balanced_datasets

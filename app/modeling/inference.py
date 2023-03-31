import torch
import transformers
from app.modeling.model import TweetClassificationModel
from transformers import AutoModel, AutoTokenizer


class Inference:
    def __init__(
        self,
        model_checkpoint: str,
        labels: list[str],
        saved_model_name: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model_checkpoint = model_checkpoint
        self.labels = labels
        self.model = TweetClassificationModel(self.model_checkpoint, len(self.labels))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.saved_model_name = saved_model_name
        self.device = device

    def make_predictions(self, text: str):
        """
        Make predictions on the given text using the pre-trained model.

        Args:
            text (str): The input text for which to make predictions.

        Returns:
            str: The predicted class label.
        """
        tokenized_input = self.tokenizer(text, return_tensors="pt")
        prediction_logits = self.model(**tokenized_input.to(self.device))
        pred_id = torch.argmax(prediction_logits.logits, dim=1)
        return self.labels[pred_id.item()]

    def prediction_pipeline(self, text: str):
        """
        Load the pre-trained model and perform predictions on the given text.

        Args:
            text (str): The input text for which to make predictions.

        Returns:
            str: The predicted class label.
        """
        self.model.load_state_dict(
            torch.load(
                f"app/modeling/models/{self.saved_model_name}.pth", map_location="cpu"
            )
        )
        self.model.to(self.device)
        return self.make_predictions(text)

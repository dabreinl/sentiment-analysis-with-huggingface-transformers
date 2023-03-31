import torch
import transformers
from torch import nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput


class TweetClassificationModel(nn.Module):
    def __init__(self, checkpoint: str, num_classes: int):
        """
        Initialize the TweetClassificationModel class, which is a custom classifier built on top of a pre-trained model.

        Args:
            checkpoint (str): The path or identifier of the pre-trained model checkpoint.
            num_classes (int): The number of classes for classification.
        """
        self.num_classes = num_classes
        self.model_dim = AutoConfig.from_pretrained(
            checkpoint
        ).hidden_size  # depending on the model config might not be dim but hidden_size
        super(TweetClassificationModel, self).__init__()
        self.distilbert_base = model = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.model_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Define the forward pass of the TweetClassificationModel.

        Args:
            input_ids (torch.Tensor): The input token ids.
            attention_mask (torch.Tensor): The attention mask for input tokens.
            labels (Optional[torch.Tensor]): The ground truth labels for the input. Defaults to None.

        Returns:
            TokenClassifierOutput: An object containing the logits, loss, hidden states, and attentions.
        """
        outputs = self.distilbert_base(
            input_ids=input_ids, attention_mask=attention_mask
        )

        last_hidden_state = outputs[0]  # only the hidden state from the last layer

        sequence_outputs = self.dropout(last_hidden_state)

        logits = self.classifier(
            sequence_outputs[:, 0, :].view(
                -1, self.model_dim
            )  # this is why argmax(dim=1) gives me the right argmax because we now have a two dimensional tensor
        )  # Only taking the hidden state of cls token

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

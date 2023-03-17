import torch
import transformers
from torch import nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput


class TweetClassificationModel(nn.Module):
    def __init__(self, checkpoint, num_classes):
        self.num_classes = num_classes

        super(TweetClassificationModel, self).__init__()
        self.distilbert_base = model = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert_base(
            input_ids=input_ids, attention_mask=attention_mask
        )

        last_hidden_state = outputs[0]  # only the hidden state from the last layer

        sequence_outputs = self.dropout(last_hidden_state)

        logits = self.linear(
            sequence_outputs[:, 0, :]
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

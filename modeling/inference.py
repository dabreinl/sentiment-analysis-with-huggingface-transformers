import torch
import transformers


def make_predictions(text: str, labels: list, tokenizer, model):
    tokenized_input = tokenizer(text, return_tensors="pt")
    prediction_logits = model(**tokenized_input.to(torch.device("mps")))
    pred_id = torch.argmax(prediction_logits.logits, dim=1)
    return labels[pred_id.item()]

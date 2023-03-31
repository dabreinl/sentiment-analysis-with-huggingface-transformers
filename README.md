# sentiment-analysis-with-huggingface-transformers

This repository just provides fine tuning of distilbert transformer on the emotions dataset containing tweets and labels ('sadness', 'joy', 'love', 'anger', 'fear', 'surprise').


The trained model is then served with a very simple frontend.

In order to use the application you should first train a model which can be done by running the training pipeline training.py .

After running the pipeline you can reference the trained model in the main.py config dictionary.

In order to then run the application you can just run docker-compose up, you should then be able to see the frontend at your localhost.
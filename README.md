# sentiment-analysis-with-huggingface-transformers 🤗

This repository provides fine tuning of distilbert transformer on the huggingface emotion dataset containing tweets and labels **('sadness', 'joy', 'love', 'anger', 'fear', 'surprise')**.


The trained model is then served with a very simple frontend.

In order to use the application you should first train a model which can be done by running the training pipeline: 
```
$python3 training.py
```
After running the pipeline you can reference the trained model in the main.py config dictionary.

In order to then run the application you can just run docker-compose up. You should now be able to see the frontend at your localhost.

# Some suggestions for further model improvement:

* After analyzing the instances with the highest loss (see visualize-model-performance notebook) it could be seen that some instances were incorrectly labeled -> by relabeling these instances, the model performance could definitely be improved

* Additional resampling methods could be experimented with to further improve model performance

* Additional learning rate schedulers could be tested to ensure that the loss function continues to converge to the minimum

* In addition to resampling methods, the loss function could be adjusted to pay more attention to observations of underrepresented classes (greater weight)
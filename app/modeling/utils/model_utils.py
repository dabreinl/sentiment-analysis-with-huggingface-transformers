import numpy as np


class EarlyStopper:
    def __init__(self, patience=0, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float):
        """
        Determine if the training should be stopped early based on the validation loss and the pre-defined patience and min_delta values.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if the training should be stopped early, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

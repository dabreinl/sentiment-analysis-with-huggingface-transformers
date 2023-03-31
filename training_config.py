class TrainingConfig:
    """
    TrainingConfig is a configuration class that contains hyperparameters and other settings
    for the training process. It centralizes the values for easy customization and management.
    """

    def __init__(
        self,
        model_checkpoint: str = "distilbert-base-uncased",
        dataset_name: str = "emotion",
        batch_size: int = 32,
        imbalanced: bool = True,
        balancer: str = "augmentation",
        load_model: bool = False,
        saved_model_name: str = "distilbert-base-finetuned-for-tweet-classification",
        model_save_name: str = "distilbert-base-finetuned-for-tweet-classification-with-augmentation-and-scheduler",
        epochs: int = 10,
        lr: float = 1e-5,
        scheduler: bool = True,
        early_stopping_patience: int = 1,
    ):
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.imbalanced = imbalanced
        self.balancer = balancer
        self.load_model = load_model
        self.saved_model_name = saved_model_name
        self.model_save_name = model_save_name
        self.epochs = epochs
        self.lr = lr
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

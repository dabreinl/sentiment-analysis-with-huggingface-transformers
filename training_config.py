class TrainingConfig:
    """
    This is the central Config Class which is used to define all the hyperparameters used in Training
    """

    def __init__(
        self,
        model_checkpoint="distilbert-base-uncased",
        dataset_name="emotion",
        batch_size=32,
        imbalanced=True,
        balancer="augmentation",
        load_model=False,
        saved_model_name="distilbert-base-finetuned-for-tweet-classification",
        model_save_name="distilbert-base-finetuned-for-tweet-classification-with-augmentation-and-scheduler",
        epochs=10,
        lr=1e-5,
        scheduler=True,
        early_stopping_patience=1,
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

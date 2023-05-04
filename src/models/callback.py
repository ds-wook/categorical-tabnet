class EarlyStoppingCallback:
    def __init__(self, min_delta: float = 0.1, patience: int = 5):
        self.min_delta = min_delta
        self.patience = patience
        self.best_epoch_score = 0

        self.attempt = 0
        self.best_score = None
        self.stop_training = False

    def __call__(self, validation_loss: float):
        self.epoch_score = validation_loss

        if self.best_epoch_score == 0:
            self.best_epoch_score = self.epoch_score

        elif self.epoch_score > self.best_epoch_score - self.min_delta:
            self.attempt += 1

            if self.attempt >= self.patience:
                self.stop_training = True

        else:
            self.best_epoch_score = self.epoch_score
            self.attempt = 0

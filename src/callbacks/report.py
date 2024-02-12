import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class ReportCallback(Callback):
    def __init__(self):
        super(ReportCallback, self).__init__()
    
    def on_test_end(self, trainer, pl_model):
        pass

    def on_fit_end(self, trainer, pl_model):
        pass
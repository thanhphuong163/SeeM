import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class VisualizationCallback(Callback):
    def __init__(self):
        super(VisualizationCallback, self).__init__()
    
    def on_test_end(self, trainer, pl_model):
        pass
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from codecarbon import EmissionsTracker

from ast_model import ASTModel

class LigthningAST(pl.LightningModule):
    def __init__(self, n_frames, n_mels, n_classes, model_size, lr, n_epochs, wandb_logger=False):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = ASTModel(label_dim=n_classes, fstride=16, tstride=16, input_tdim=n_frames, input_fdim=n_mels, model_size=model_size)
        self.model_size = model_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.wandb_logger = wandb_logger


        self.epoch_val_loss_evidence = []
        self.epoch_val_acc_evidece = []
        self.val_loss_evidence = []
        self.val_acc_evidence = []
        self.tracker = None
        self.save_hyperparameters(ignore=['model', 'wandb_logger'])

        # attributes for saving test predctions and test_targets
        self.test_preds = torch.tensor([])
        self.test_targets = torch.tensor([]) 

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # print(f"OPTIMIZER IS SET WITH INITAL LEARNING RATE THAT IS EQUAL TO {self.lr}")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, verbose=True)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss"
            }
        }


    def training_step(self, batch, batch_idx):
        loss, accuracy = self._get_preds_loss_accuracy(batch)
        self.log_dict({
            "train_loss": loss,
            "train_accuracy": accuracy,
        }, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, val_accuracy = self._get_preds_loss_accuracy(batch)
        self.epoch_val_loss_evidence.append(val_loss)
        self.epoch_val_acc_evidece.append(val_accuracy)

        self.log_dict({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        return val_loss

    def on_train_epoch_start(self):
        # start tracking emissions for current epoch
        self.tracker = EmissionsTracker()
        self.tracker.start()

    def on_train_epoch_end(self):
        # end tracking of emission of current epoch
        if self.tracker:
            emissions = self.tracker.stop()
            print(f"EMISSIONS FROM CURRENT EPOCH: {emissions}")
            self.log("CO2eq", emissions, rank_zero_only=True)
            self.tracker = None

    def on_test_epoch_start(self):
        # print("RESETING THE EVIDENCE TENSORS ON THE TEST EPOCH START!!!")
        self.test_preds = torch.tensor([])
        self.test_targets = torch.tensor([])

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.forward(inputs)
        self.test_preds = torch.cat((self.test_preds, torch.argmax(preds, dim=-1).detach().cpu()))
        self.test_targets = torch.cat((self.test_targets, targets.detach().cpu()))
        return

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.forward(inputs)
        return targets, torch.argmax(preds, dim=-1)

    def _get_preds_loss_accuracy(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        return loss, accuracy

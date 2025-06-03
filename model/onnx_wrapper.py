import torch
import pytorch_lightning as pl


class CLIPReIDONNXWrapper(pl.LightningModule):
    def __init__(self, model, use_camera=False, use_view=False):
        super().__init__()
        self.model = model
        self.use_camera = use_camera
        self.use_view = use_view
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x, cam_label=None, view_label=None):
        self.model.eval()
        
        with torch.no_grad():
            if self.use_camera and self.use_view:
                if cam_label is None or view_label is None:
                    raise ValueError("Camera and view labels are required for this model")
                features = self.model(x, cam_label=cam_label, view_label=view_label)
            elif self.use_camera:
                if cam_label is None:
                    raise ValueError("Camera label is required for this model")
                features = self.model(x, cam_label=cam_label)
            elif self.use_view:
                if view_label is None:
                    raise ValueError("View label is required for this model")
                features = self.model(x, view_label=view_label)
            else:
                features = self.model(x)
                
        return features
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch[0]
        cam_label = batch[2] if len(batch) > 2 and self.use_camera else None
        view_label = batch[4] if len(batch) > 4 and self.use_view else None
        
        return self.forward(x, cam_label, view_label)
    
    def configure_optimizers(self):
        return None
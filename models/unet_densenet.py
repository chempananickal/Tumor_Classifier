import torch
import torch.nn as nn
from torchvision import models

# Default class ordering (used if checkpoint does not embed its own ordering).
# Training now saves the dataset's actual ordering inside the checkpoint as 'classes'.
CLASSES = ('glioma', 'meningioma', 'negative', 'pituitary') # NOTE Fixed by AI
#CLASSES = ("pituitary", "glioma", "meningioma", "negative") was what it used to look like. Hilariouly screwed up the inference - James
NUM_CLASSES = len(CLASSES)


def get_model(pretrained: bool = True) -> nn.Module:
    """Return a DenseNet121-based classifier for 4 tumor classes.

    For now we use plain DenseNet121 classifier head adaptation
    because the accuracy seems more than good enough.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    return model


class ModelWithHooks(nn.Module): # NOTE: AI Generated
    """Wrapper to expose a specific feature layer for Grad-CAM.

    We target the last denseblock's norm (model.features.denseblock4) output before pooling.
    """
    def __init__(self, model: nn.Module, target_layer_name: str = "features.denseblock4"):
        super().__init__()
        self.model = model
        self.target_layer_name = target_layer_name
        self._activation = None
        self._gradient = None
        # Register hook
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _get_target_layer(self):
        module = self.model
        for attr in self.target_layer_name.split('.'):
            module = getattr(module, attr)
        return module

    def _forward_hook(self, module, inp, out):
        self._activation = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradient = grad_output[0].detach()

    @property
    def activation(self):
        return self._activation

    @property
    def gradient(self):
        return self._gradient

    def forward(self, x):
        return self.model(x)


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: str = "cpu") -> nn.Module:
    """Load model weights from a checkpoint file, handling various serialization formats."""
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and 'state_dict' in state:
        sd = state['state_dict']
        # Strip potential "model." prefixes
        new_sd = {k.split('model.', 1)[-1] if k.startswith('model.') else k: v for k, v in sd.items()}
        model.load_state_dict(new_sd)
    else:
        model.load_state_dict(state)
    model.eval()
    return model

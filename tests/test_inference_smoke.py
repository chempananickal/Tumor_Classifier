import torch
from models.unet_densenet import get_model, CLASSES

def test_forward_shape():
    model = get_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, len(CLASSES))

# NOTE: AI Generated
# I don't know what the AI was thinking here
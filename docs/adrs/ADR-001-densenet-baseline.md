# ADR-001: DenseNet121 as Initial Backbone

## Status
Accepted

## Context
A baseline convolutional neural network was required for 4-class brain MRI tumor slice classification. Priorities: (1) strong performance on moderate-sized datasets, (2) compatibility with Grad-CAM, (3) availability of stable pretrained weights.

## Decision
Use `torchvision.models.densenet121` with ImageNet weights and replace classifier head with a 4-logit linear layer.

## Rationale
- Dense connectivity promotes feature reuse and stable gradients.
- Widely evaluated in medical imaging literature—lower risk for unexpected behaviors.
- Straightforward integration with hook-based Grad-CAM (clear terminal convolutional feature block).
- Parameter count moderate vs larger ResNet variants.

## Consequences
- Limits ability to leverage very long-range spatial relationships (compared to ViT/Transformers).
- May underperform newer ConvNeXt/EfficientNet variants on larger datasets.
- Facilitates rapid experimentation; refactors can introduce pluggable backbones later.

## Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| ResNet50 | Stable, common | Slightly larger, CAM quality similar |
| EfficientNet-B3 | Better accuracy/params | Slightly more brittle LR tuning, CAM less sharp |
| ConvNeXt-T | Modern redesign | Additional complexity, not initially implemented |
| ViT-B/16 | Global context | Requires more data/pretraining for stability |

## Future Evolution
Introduce architecture flag for alternate backbones; update Grad-CAM hook target accordingly.

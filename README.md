## Branch Description
The code here is supplementary/experimental. It contains three Patient-Level self attention models. (1) multi-modal: combines lesion-level and patient-level loss; (2) instance-based pooling: self-attention pooling module (gated or not gated) individual logits from a frozen resnet101 pretrained on ISIC 2020 dataset; (3) embedding-based pooling: unfrozen layer 4 and head of resnet101 outputs an embedding (dim = 500) that is processed by a self-attention pooling module (gated or not gated).


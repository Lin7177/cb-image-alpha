"""I20 paper CNN: three convolution blocks and a two-logit head."""
import torch
from torch import Tensor, nn

MODEL_VERSION = "paper-i20-v1"


class PaperCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks = []
        for i, (incoming, outgoing) in enumerate(((1, 64), (64, 128), (128, 256))):
            convolution = nn.Conv2d(incoming, outgoing, (5, 3),
                                    stride=(3, 1) if i == 0 else (1, 1),
                                    padding=(2, 1),
                                    dilation=(2, 1) if i == 0 else (1, 1))
            nn.init.xavier_uniform_(convolution.weight)
            # Conv bias retains PyTorch's default uniform initialization, as in reference.
            blocks.append(nn.Sequential(convolution, nn.BatchNorm2d(outgoing),
                                        nn.LeakyReLU(negative_slope=0.01),
                                        nn.MaxPool2d((2, 1), ceil_mode=True)))
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(256 * 3 * 60, 2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.01)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 64, 60):
            raise ValueError("Expected [N,1,64,60] I20 images")
        x = images
        for block in self.blocks:
            x = block(x)
        return self.head(self.dropout(torch.flatten(x, 1)))
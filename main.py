from models.model import EffUnet
import torch
from torchinfo import summary

if __name__ == "__main__":

    batch_size = 2
    mock_data = torch.rand((batch_size, 3, 224, 224), dtype=torch.float32)
    model = EffUnet(0)
    x = model(mock_data)

    summary(model, input_size=(batch_size, 3, 224, 224))
    print(x.shape)

    input_names = ["input_image"]
    output_names = ["output_logits"]

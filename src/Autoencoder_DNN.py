import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),

            torch.nn.Linear(2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_shape),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        inp_shape = x.shape
        x = torch.flatten(x, 1)
        x = self.seq(x)
        return x.reshape(inp_shape)

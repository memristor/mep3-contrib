import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection
        shortcut = self.shortcut(x)

        # Add the residual and apply ReLU
        out += shortcut
        out = self.relu(out)

        return out


class CNNTracking5(nn.Module):
    def __init__(self, num_residual_blocks1=2, num_residual_blocks2=12, num_residual_blocks3=5):
        super(CNNTracking5, self).__init__()

        filter_count = 32
        self.conv1 = nn.Conv2d(3, filter_count, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks1 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks1)
        self.mid_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_blocks2 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks2)

        self.conv_seg = nn.Conv2d(filter_count, 1, kernel_size=3, stride=(1, 1), padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_1 = nn.Conv2d(filter_count, filter_count, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(filter_count, filter_count, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(filter_count, filter_count, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.res_blocks3 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks3)

        self.conv_coord = nn.Conv2d(filter_count, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.coord_out_res = (25, 37)  # W, H


    def _make_residual_blocks(self, in_channels, out_channels, num_blocks):
        blocks = []
        for i in range(num_blocks):
            # stride = 2 if i == 0 else 1  # Use stride 2 for the first block
            stride = 1
            blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels  # Update in_channels for the next block
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.res_blocks1(out)
        segmentation = self.conv_seg(out)
        out = self.mid_pool1(out)
        out = self.res_blocks2(out)

        out = self.pool1(self.relu(self.conv1_1(out)))
        out = self.pool2(self.relu(self.conv1_2(out)))
        #out = self.pool3(self.relu(self.conv1_3(out)))

        out = self.res_blocks3(out)

        coords = self.conv_coord(out)

        return segmentation, coords

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.dropout1 = nn.Dropout2d(p=dropout_rate)  # Dropout layer after first convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)  # Dropout layer after second convolution

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
        #out = self.dropout1(out)  # Apply dropout after first convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)  # Apply dropout after second convolution

        # Shortcut connection
        shortcut = self.shortcut(x)

        # Add the residual and apply ReLU
        out += shortcut
        out = self.relu(out)

        return out


class CNNTracking7(nn.Module):
    def __init__(self, num_residual_blocks1=2, num_residual_blocks2=12, num_residual_blocks3=5,
                 num_residual_blocks_branch2=6, dropout1=0.5, dropout2=0.1):
        super(CNNTracking7, self).__init__()

        filter_count = 32
        self.conv1 = nn.Conv2d(3, filter_count, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks1 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks1, False, dropout1)
        self.mid_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_blocks2 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks2, False, dropout1)

        self.conv_seg = nn.Conv2d(filter_count, 1, kernel_size=3, stride=(1, 1), padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_1 = nn.Conv2d(filter_count, filter_count, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.res_blocks3 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks3, False, dropout2)

        self.conv_coord = nn.Conv2d(filter_count, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.res_blocks2_1 = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks_branch2 // 3, True, dropout1)
        self.res_blocks2_2 = self._make_residual_blocks(filter_count, filter_count * 2, num_residual_blocks_branch2 // 3, True, dropout1)
        self.res_blocks2_3 = self._make_residual_blocks(filter_count * 2, filter_count * 4, num_residual_blocks_branch2 // 3, True, dropout1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 7 * filter_count * 4, 7 * 6)
        self.dropout = nn.Dropout(dropout1)

        self.coord_out_res = (37, 50)  # W, H

    def _make_residual_blocks(self, in_channels, out_channels, num_blocks, shrink_first=False, dropout=0.0):
        blocks = []
        for i in range(num_blocks):
            if shrink_first:
                stride = 2 if i == 0 else 1  # Use stride 2 for the first block
            else:
                stride = 1
            blocks.append(ResidualBlock(in_channels, out_channels, stride=stride, dropout_rate=dropout))
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

        branching_out = self.pool1(self.relu(self.conv1_1(out)))

        out = self.res_blocks3(branching_out)

        coords = self.conv_coord(out)

        branch2_out = self.res_blocks2_1(branching_out)
        probe1 = branch2_out
        branch2_out = self.res_blocks2_2(branch2_out)
        branch2_out = self.res_blocks2_3(branch2_out)

        branch2_out = self.flatten(branch2_out)
        branch2_out = self.dropout(branch2_out)
        area_occupancy = self.fc1(branch2_out)

        return segmentation, coords, area_occupancy, probe1

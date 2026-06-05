import torch.nn as nn
from model.layers import Conv2dWithConstraint


class EEGNet(nn.Module):
    def __init__(self,
                num_channels: int,
                F1=8, D=2, F2= 'auto', T1= 100, T2=30, P1=4, P2=8, pool_mode= 'mean',
                drop_out=0.25): # T1 is half sampling rate
        super(EEGNet, self).__init__()
    
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1),  padding=(0, T1//2), bias=False),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1), stride=4),
            nn.Dropout(drop_out)
        )
        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2),  padding=(0, T2//2), groups=F1 * D),
            nn.Conv2d(F2, F2, 1,  stride=1, bias=False, padding=0),
            nn.BatchNorm2d(F2),
            # ActSquare(),
            nn.ELU(),
            pooling_layer((1, P2), stride=8),
            # ActLog(),
            nn.Dropout(drop_out)
        )
        self.flatten = nn.Flatten()
        # self.bn = nn.BatchNorm2d(F2)


    def forward(self, x):
        spectral_features = self.spectral(x)
        spatial_features = self.spatial(spectral_features)
        temporal_features = self.temporal(spatial_features)
        output = self.flatten(temporal_features)
        return spatial_features, temporal_features, output


class Classifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Classifier, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_features, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)
        return x


class Net(nn.Module):

    def __init__(self,
                 num_classes: int,
                 num_channels: int,
                 sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = EEGNet(num_channels=num_channels)

        input_features = 288
        self.classifier = Classifier(input_features, num_classes)

    def forward(self, x, intermediate=False):

        spatial_features, temporal_features, features = self.backbone(x)
        output = self.classifier(features)
        if intermediate:  # If intermediate is true, return the local and global features
            return features, output
        else:
            return output


def get_teacher_model(args):
    model = Net(num_classes=args.num_classes,
                num_channels=args.num_channels,
                sampling_rate=args.sampling_rate)
    return model


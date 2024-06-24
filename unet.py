import torch
import torch.nn as nn

class UpConvolution(nn.Module):
    """" Up sampling and convolution block """
    def __init__(self, in_channels, out_channels, num_classes=1, final_conv=False):
        super(UpConvolution, self).__init__()
        self.final_conv = final_conv
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.conv_up = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, \
                                       kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                      kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, \
                                      kernel_size=3, padding=1, padding_mode='reflect')
        self.final_convolution = torch.nn.Conv2d(in_channels=out_channels, out_channels=num_classes, \
                                      kernel_size=3, padding=1, padding_mode='reflect')
        self.dropout = torch.nn.Dropout(p=0.3, inplace=False)

    def forward(self, x, x_from_ds):
        x = self.upsample(x)
        x = self.conv_up(x)
        x = self.dropout(x)
        x = torch.cat((x, x_from_ds), dim=1)
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        if self.final_conv:
            x = self.final_convolution(x)
        return x
    
class DownSampling(nn.Module):
    """ Double convolution of the downsampling part """
    def __init__(self, in_channels, out_channels, use_maxpool=True):
        super(DownSampling, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                      kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, \
                                      kernel_size=3, padding=1, padding_mode='reflect')
        self.ReLu = torch.nn.ReLU()
        self.MaxPool = torch.nn.MaxPool2d(kernel_size=2)
        self.use_maxpool = use_maxpool
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.ReLu(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.ReLu(x)
        x = self.dropout(x)
        if not self.use_maxpool:
            return x
        else:
            x_skip = x
            x = self.MaxPool(x)
            return x, x_skip
        
class Unet(nn.Module):
    """ Unet from previous steps """
    def __init__(self, in_layers, num_classes):
        super(Unet, self).__init__()
        self.conv_1 = DownSampling(in_channels=in_layers, out_channels=64)
        self.conv_2 = DownSampling(in_channels=64, out_channels=128)
        self.conv_3 = DownSampling(in_channels=128, out_channels=256)
        self.conv_4 = DownSampling(in_channels=256, out_channels=512)
        self.conv_5 = DownSampling(in_channels=512, out_channels=1024, use_maxpool=False)
        
        self.up_conv_1 = UpConvolution(in_channels=1024, out_channels=512)
        self.up_conv_2 = UpConvolution(in_channels=512, out_channels=256)
        self.up_conv_3 = UpConvolution(in_channels=256, out_channels=128)
        self.up_conv_4 = UpConvolution(in_channels=128, out_channels=64, num_classes=num_classes, final_conv=True)

    def forward(self, x):
        # Downsampling part
        x, skip1 = self.conv_1.forward(x)
        x, skip2 = self.conv_2.forward(x)
        x, skip3 = self.conv_3.forward(x)
        x, skip4 = self.conv_4.forward(x)
        x = self.conv_5.forward(x)

        # Upsampling part
        x = self.up_conv_1.forward(x, skip4)
        x = self.up_conv_2.forward(x, skip3)
        x = self.up_conv_3.forward(x, skip2)
        x = self.up_conv_4.forward(x, skip1)

        return x

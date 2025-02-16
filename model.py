# model.py - PyTorch Version
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

'''
CNN-AE
'''


# Use 64x64 or 256x256


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64->32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32->16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16->8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 8->16,
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16->32,
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 32->64,
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


'''
FC-AE
'''


# Use 64x64


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder: 40000 (200*200) ==> 16384 (128*128)
        self.encoder = nn.Sequential(
            nn.Flatten(),  # reshaping it into a 1-d vector
            torch.nn.Linear(64 * 64, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )

        # Decoder: 16384 (128*128) ==> 40000 (200*200)
        self.decoder = nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 64 * 64),
            torch.nn.Sigmoid()  # Sigmoid to normalize output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reshape the output to the desired output size (128x128). Assuming the input x has a shape [N, C, H, W]
        x = decoded.view(-1, 1, 64, 64)
        return x


'''
U-Net
'''


# Use 32x32 64x64 or 256x256
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the encoder part
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Define the decoder part
        self.decoder4 = self.conv_transpose_block(512, 256)
        self.up4 = self.conv_block(512, 256)
        self.decoder3 = self.conv_transpose_block(256, 128)  # After concat
        self.up3 = self.conv_block(256, 128)
        self.decoder2 = self.conv_transpose_block(128, 64)  # After concat
        self.up2 = self.conv_block(128, 64)
        # self.decoder1 = self.conv_transpose_block(64, 32)  # After concat

        # Define final convolution
        self.up1 = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            # nn.Dropout(0.5)
        )
        return block

    def conv_transpose_block(self, in_channels, out_channels):
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return up

    def forward(self, x1):
        # Encoder
        enc1 = self.encoder1(x1)  # Ch=1->64
        x2 = self.Maxpool(enc1)
        enc2 = self.encoder2(x2)  # 64->128, Scale=64x64->32x32
        x3 = self.Maxpool(enc2)
        enc3 = self.encoder3(x3)  # 128->256, Scale=32->16
        x4 = self.Maxpool(enc3)
        enc4 = self.encoder4(x4)  # 256->512, Scale=16->8

        # Decoder
        dec4 = self.decoder4(enc4)  # 512->256, 8->16
        up4 = self.up4(torch.cat([dec4, enc3], dim=1))  # 256+256=512->256
        dec3 = self.decoder3(up4)  # 256->128, 16->32
        up3 = self.up3(torch.cat([dec3, enc2], dim=1))  # 128+128=256->128
        dec2 = self.decoder2(up3)  # 128->64, 32->64
        up2 = self.up2(torch.cat([dec2, enc1], dim=1))  # 64+64=128->64
        up1 = self.up1(up2)  # 64->1

        # Final convolution
        out = F.sigmoid(up1)
        return out

'''
AE-SNN
'''


class AE_SNN(nn.Module):
    def __init__(self):
        super(AE_SNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # size 64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 64x64 -> 32x32
            nn.Dropout(0.25),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # size 32x32
            nn.ReLU(),
            nn.Flatten(),

            # Dense layers with SELU activations
            nn.BatchNorm1d(32 * 32 * 32),
            nn.Linear(32 * 32 * 32, 1024),
            nn.SELU(),

            nn.Dropout(0.25),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4096),
            nn.SELU(),

            nn.Dropout(0.25),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 32 * 32 * 32),
            nn.SELU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 32, 32)),  # Reshape to 32 channels of 32x32

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # UpSampling to 16x64x64
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # Upsampling to 1x64x64
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode the input image to a feature vector
        encoded = self.encoder(x)
        # Decode the feature vector back to an image
        decoded = self.decoder(encoded)
        return decoded


'''
Attention U-Net
'''


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):  # ex:256,256,128
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Att_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(Att_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # s=64x64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # s=32x32

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # s=16x16

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # s=8x8

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # # decoding + concat path
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)  # s=16x16
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d1 = self.final_act(d1)
        return d1


'''
HPM_U-Net
'''

class HPM_Block(nn.Module):
    def __init__(self, ch_in, ch_out):  # ex ch_in=64, ch_out=64
        super(HPM_Block, self).__init__()

        # First 3x3 Convolution + Batch Normalization + ReLU
        self.conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.relu1 = nn.ReLU(inplace=True)

        # Second 3x3 Convolution + Batch Normalization + ReLU with downsampling
        self.conv2 = nn.Conv2d(ch_in, int(ch_in / 2), kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(ch_in / 2))
        self.relu2 = nn.ReLU(inplace=True)

        # Third 3x3 Convolution + Batch Normalization + ReLU with downsampling
        self.conv3 = nn.Conv2d(int(ch_in / 2), int(ch_in / 2), kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(int(ch_in / 2))
        self.relu3 = nn.ReLU(inplace=True)

        # 1x1 Convolution for final output
        self.conv4 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        # Apply the first 3x3 Convolution path
        x1 = self.relu1(self.bn1(self.conv1(x)))  # x1->64

        # Apply the second 3x3 Convolution path
        x2 = self.relu2(self.bn2(self.conv2(x1)))  # x2->32

        # Apply the third 3x3 Convolution path
        x3 = self.relu3(self.bn3(self.conv3(x2)))  # x3->32
        # print(x1.shape,x2.shape,x3.shape)
        # Concatenate the outputs
        x_cat = torch.cat([x1, x2, x3], dim=1)  # x_cat->128

        # Apply the 1x1 Convolution + BatchNorm
        shortcut = self.bn4(self.conv4(x))  # shortcut->128

        # Element-wise sum with the original x
        x_out = x_cat + shortcut

        return x_out


class HPM_Attention_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(HPM_Attention_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = HPM_Block(ch_in=64, ch_out=128)
        self.Conv3 = HPM_Block(ch_in=128, ch_out=256)
        self.Conv4 = HPM_Block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)  # 64x64x64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # 32x32x128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 16x16x256

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 8x8x512

        # Decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d1 = self.final_act(d1)
        return d1

'''

class HPM_Block(nn.Module):
    def __init__(self, ch_in, ch_out):  # ex ch_in=64, ch_out=64
        super(HPM_Block, self).__init__()

        # First 3x3 Convolution + Batch Normalization + ReLU
        self.conv1 = nn.Conv2d(ch_in, int(ch_in / 2), kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(int(ch_in / 2))
        self.relu1 = nn.ReLU(inplace=True)

        # Second 3x3 Convolution + Batch Normalization + ReLU with downsampling
        self.conv2 = nn.Conv2d(int(ch_in / 2), int(ch_in / 4), kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(ch_in / 4))
        self.relu2 = nn.ReLU(inplace=True)

        # Third 3x3 Convolution + Batch Normalization + ReLU with downsampling
        self.conv3 = nn.Conv2d(int(ch_in / 4), int(ch_in / 4), kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(int(ch_in / 4))
        self.relu3 = nn.ReLU(inplace=True)

        # 1x1 Convolution for final output
        self.conv4 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        # Apply the first 3x3 Convolution path
        x1 = self.relu1(self.bn1(self.conv1(x)))  # x1->64

        # Apply the second 3x3 Convolution path
        x2 = self.relu2(self.bn2(self.conv2(x1)))  # x2->32

        # Apply the third 3x3 Convolution path
        x3 = self.relu3(self.bn3(self.conv3(x2)))  # x3->32
        # print(x1.shape,x2.shape,x3.shape)
        # Concatenate the outputs
        x_cat = torch.cat([x1, x2, x3], dim=1)  # x_cat->128

        # Apply the 1x1 Convolution + BatchNorm
        shortcut = self.bn4(self.conv4(x))  # shortcut->128

        # Element-wise sum with the original x
        x_out = x_cat + shortcut

        return x_out


class HPM_Attention_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(HPM_Attention_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)

        self.Conv2 = HPM_Block(ch_in=64, ch_out=64)
        self.Conv2_1 = conv_block(ch_in=64, ch_out=128)

        self.Conv3 = HPM_Block(ch_in=128, ch_out=128)
        self.Conv3_1 = conv_block(ch_in=128, ch_out=256)

        self.Conv4 = HPM_Block(ch_in=256, ch_out=256)
        self.Conv4_1 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)  # 64x64x64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # 32x32x64
        x2 = self.Conv2_1(x2)  # 32x32x128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 16x16
        x3 = self.Conv3_1(x3)  # 16x16x256

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 8x8
        x4 = self.Conv4_1(x4)  # 8x8x512

        # Decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d1 = self.final_act(d1)

        return d1

'''

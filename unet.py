'''
Defines the UNet model and its parts/possible modifications
Contains options to use/not use modifications described on the paper
(residual connections, bias, batch_norm etc)

Author: Diedre Carmo
https://github.com/dscarmo
'''
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F



'''
Following functions abstract 3d and 2d unet
if dim == '2d':
elif dim == '3d':
'''
def assert_dim(dim):
    assert dim in ('2d', '3d'), "dim {} not supported".format(dim)

def conv(in_ch, out_ch, kernel_size, padding, bias, dim='2d'):
    assert_dim(dim)
    if dim == '2d': return nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias)
    elif dim == '3d': return nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=bias)

def batch_norm(out_ch, dim='2d'):
    assert_dim(dim)
    if dim == '2d': return nn.BatchNorm2d(out_ch)
    elif dim == '3d': return nn.BatchNorm3d(out_ch)

def max_pool(kernel_size, dim='2d'):
    assert_dim(dim)
    if dim == '2d': return nn.MaxPool2d(kernel_size)
    elif dim == '3d': return nn.MaxPool3d(kernel_size)

def conv_transpose(in_ch, out_ch, kernel_size, stride, bias, dim='2d'):
    assert_dim(dim)
    if dim == '2d': return nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size, stride=stride, bias=bias)
    elif dim == '3d': return nn.ConvTranspose3d(in_ch//2, in_ch//2, kernel_size, stride=stride, bias=bias)

class double_conv(nn.Module):
    '''
    (conv => BN => ReLU) * 2, one UNET Block
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(double_conv, self).__init__()
        if bn:
            self.conv = nn.Sequential(
                conv(in_ch, out_ch, 3, 1, bias, dim=dim),
                batch_norm(out_ch, dim=dim),
                nn.LeakyReLU(inplace=True),
                conv(out_ch, out_ch, 3, 1, bias, dim=dim),
                batch_norm(out_ch, dim=dim),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                conv(in_ch, out_ch, 3, 1, bias, dim=dim),
                nn.LeakyReLU(inplace=True),
                conv(out_ch, out_ch, 3, 1, bias, dim=dim),
                nn.LeakyReLU(inplace=True)
            )
        self.residual = residual
        if residual:
            self.residual_connection = conv(in_ch, out_ch, 1, 0, bias, dim=dim)

    def forward(self, x):
        y = self.conv(x)
        if self.residual:
            return y + self.residual_connection(x)
        else:
            return y
        


class inconv(nn.Module):
    '''
    Input convolution
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, residual=residual, bias=bias, bn=bn, dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''
    Downsample conv
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            max_pool(2, dim=dim),
            double_conv(in_ch, out_ch, residual=residual, bias=bias, bn=bn, dim=dim)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    '''
    Upsample conv
    '''
    def __init__(self, in_ch, out_ch, residual=False, bias=False, bn=True, dim='2d'):
        super(up, self).__init__()
        self.up = conv_transpose(in_ch, in_ch, 2, 2, bias=bias, dim=dim)
        self.conv = double_conv(in_ch, out_ch, residual, bias=bias, bn=bn, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    '''
    Output convolution
    '''
    def __init__(self, in_ch, out_ch, bias=False, dim='2d'):
        super(outconv, self).__init__()
        self.conv = conv(in_ch, out_ch, 1, 0, bias, dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    '''
    Main model class
    '''
    @staticmethod
    def forward_test(inpt, n_channels, n_classes, apply_sigmoid, residual, small, bias, bn, dim='2d'):
        '''
        UNet unit test
        '''
        unet = UNet(n_channels, n_classes, apply_sigmoid=apply_sigmoid, residual=residual, small=small, bias=bias, bn=bn, dim=dim)
        print(unet)
        return unet.forward(inpt)
    
    def __init__(self, n_channels, n_classes, apply_sigmoid=True, residual=False, small=False, bias=False, bn=True, verbose=True, dim='2d'):
        super(UNet, self).__init__()
        big = not small
        self.inc = inconv(n_channels, 64, residual, bias, bn, dim=dim)
        self.down1 = down(64, 128, residual, bias, bn, dim=dim)
        self.down2 = down(128, 128+big*128, residual, bias, bn, dim=dim)
        if not small:
            self.down3 = down(256, 512, residual, bias, bn, dim=dim)
            self.down4 = down(512, 512, residual, bias, bn, dim=dim)
            self.up1 = up(1024, 256, residual, bias, bn, dim=dim)
            self.up2 = up(512, 128, residual, bias, bn, dim=dim)
        self.up3 = up(256, 64, residual, bias, bn, dim=dim)
        self.up4 = up(128, 64, residual, bias, bn, dim=dim)
        self.outc = outconv(64, n_classes, bias, dim=dim)
        self.apply_sigmoid = apply_sigmoid
        self.small = small
        if verbose: print("UNet using sigmoid: {} residual connections: {} small: {} number of channels: {} bias: {} batch_norm: {} dim: {}".format(self.apply_sigmoid, residual, small, n_channels, bias, bn, dim))

    def forward(self, x):
        '''
        Saves every downstep output to use in upsteps concatenation
        '''
        if self.small: # REMOVE 2 LAST DOWN AND FIRST UP
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
        else: 
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
        if self.apply_sigmoid:
            x = x.sigmoid()
        return x

def init_weights(vgg, model):
    '''
    Returns updated state dict to be loaded into model pre training with vgg weights
    '''
    state_dict = model.state_dict()
    for vk, vv in vgg.state_dict().items():
        if vk.split('.')[-1] == "weight" and vk.split('.')[0] != "classifier":
            for uk, uv in state_dict.items():
                if uk.split('.')[-1] == "weight" and uk.split('.')[0][:2] != "up":
                    if vv.shape == uv.shape:
                        print("Found compatible layer...")
                        print("VGG Key: {}".format(vk))
                        print("UNET Key: {}".format(uk))
                        print("VGG shape: {}".format(vv.shape))
                        print("UNET shape: {}".format(uv.shape))
                        state_dict[uk] = vv
                        print("Weights transfered. Check:", end=' ')
                        print(((state_dict[uk] == vgg.state_dict()[vk]).sum()/len(state_dict[uk].view(-1))).item() == 1) # one liner to check if all weights are the same
                        print("-"*20)
                        break
    return state_dict

def random_unet_test(device, dim='2d'):   
    from utils import viewnii
    
    print("Testing {} UNet".format(dim))

    assert_dim(dim)
    if dim == '2d':
        size = (1, 3, 32, 32)
    elif dim == '3d':
        size = (1, 1, 32, 32, 32)
    inpt = torch.rand(size, requires_grad=True)
    
    if dim == '2d':
        plt.subplot(1, 2, 1)
        plt.title("Random input")
        plt.imshow(inpt[:, 1].squeeze().detach().numpy(), cmap='gray')
    elif dim == '3d':
        viewnii(inpt[0, 0].squeeze().detach().numpy())
    
    begin = time.time()
    if dim == '2d':
        outpt = UNet.forward_test(inpt, 3, 1, True, True, False, False, True)
    elif dim == '3d':
        outpt = UNet.forward_test(inpt, 1, 1, True, False, True, False, False, '3d')
    print("Foward time: " + str(round(time.time() - begin, 3)) + "s")
    if dim == '2d':
        plt.subplot(1, 2, 2)
        plt.title("UNet output")
        plt.imshow(outpt.squeeze().detach().numpy(), cmap='gray')
        print(outpt.shape)
        plt.show()
    elif dim == '3d':
        viewnii(outpt.squeeze().detach().numpy(), wait=0)

def main():
    '''
    Runs if the module is called as a script (python3 unet.py)
    '''
    from sys import argv
    from utils import get_device

    print("unet module running as script, executing random unet forward test")
    if len(argv) >= 1:
        if argv[1] == '3d':
            random_unet_test(get_device(), dim='3d')
        else:
            print("{} not understood".format(argv[1]))
    else:
        random_unet_test(get_device())


if __name__ == "__main__":
    main()
    
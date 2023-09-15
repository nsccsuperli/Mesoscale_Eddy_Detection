import torch
import torch.nn.functional as F
from torch import nn

from network import Resnet
from network.wider_resnet import wider_resnet38_a2

from network.mynn import initialize_weights, Norm2d
from torch.autograd import Variable

from my_functionals import GatedSpatialConv as gsc

import cv2
import numpy as np

class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices).long()
            x = x.index_select(axis, Variable(indices))
        return x


class MyIdentity(nn.Module):
    def __init__(self, axis, offset):
        super(MyIdentity, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """

        :param x: input layer
        :param ref: reference usually data in
        :return:
        """
        return x

class SideOutputCrop(nn.Module):
    """
    This is the original implementation ConvTranspose2d (fixed) and crops
    """

    def __init__(self, num_output, kernel_sz=None, stride=None, upconv_pad=0, do_crops=True):
        super(SideOutputCrop, self).__init__()
        self._do_crops = do_crops
        self.conv = nn.Conv2d(num_output, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(1, out_channels=1, kernel_size=kernel_sz, stride=stride,
                                                padding=upconv_pad,
                                                bias=False)
            ##doing crops
            if self._do_crops:
                self.crops = Crop(2, offset=kernel_sz // 4)
            else:
                self.crops = MyIdentity(None, None)
        else:
            self.upsample = False

    def forward(self, res, reference=None):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)
            side_output = self.crops(side_output, reference)

        return side_output


class _AtrousSpatialPyramidPoolingModule(nn.Module):

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
         

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear',align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel=2, out_channel=64):
        super(ConvBlock, self).__init__()

        self.layer = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class MFED(nn.Module):


    def __init__(self, num_classes, trunk=None):
        
        super(MFED, self).__init__()
        # self.criterion = criterion
        self.num_classes = num_classes

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)

        wide_resnet = wide_resnet.module
        self.mod1 = ConvBlock()
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = F.interpolate
        del wide_resnet

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn7 = nn.Conv2d(4096, 1, 1)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
         
        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False),
            nn.Tanh())

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.final_seg)

    def forward(self, inp,input2,edge, gts=None):
        # 2,3,120,101 -------------------->2,1,120,101
        x_size = inp.size()
        output = torch.cat((inp, input2), dim=1)

        m1 = self.mod1(output)

        m2 = self.mod2(self.pool2(m1))

        m3 = self.mod3(self.pool3(m2))

        m4 = self.mod4(m3)
        m5 = self.mod5(m4)
        m6 = self.mod6(m5)
        m7 = self.mod7(m6) 
        # [ 2,1,120,101]                     1
        s3 = F.interpolate(self.dsn3(m3), x_size[2:],
                            mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(m4), x_size[2:],
                            mode='bilinear', align_corners=True) # [ 2,1,120,101]
        s7 = F.interpolate(self.dsn7(m7), x_size[2:],
                            mode='bilinear', align_corners=True)# [ 2,1,120,101]
        # # [ 2,64,120,101]
        m1f = F.interpolate(m1, x_size[2:], mode='bilinear', align_corners=True)
        #  eddy canny detection
        im_arr = inp.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()# [ 2,1,120,101]
        # given contour
        canny_edge =canny
        cs = self.res1(m1f)#[ 2,64,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d1(cs)#[ 2,32,120,101]
        cs = self.gate1(cs, s3)
        cs = self.res2(cs)  #[ 2,32,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True) #[ 2,32,120,101]
        cs = self.d2(cs) #[ 2,16,120,101]
        cs = self.gate2(cs, s4)#[ 2,16,120,101] [ 2,1,120,101] --------------->[ 2,16,120,101]
        cs = self.res3(cs)#[ 2,16,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)#[ 2,16,120,101]
        cs = self.d3(cs)#[ 2,8,120,101]
        cs = self.gate3(cs, s7)#[ 2,8,120,101] [ 2,1,120,101] --------------->[ 2,8,120,101]
        cs = self.fuse(cs)#[ 2,1,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)#[ 2,1,120,101]
        edge_out = self.sigmoid(cs)# [ 2,1,120,101]
        cat = torch.cat((edge_out, canny_edge), dim=1)#[ 2,2,120,101]
        acts = self.cw(cat)# [ 2,1,120,101]
        acts = self.sigmoid(acts) # [ 2,1,120,101]
        # output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        # aspp
        x = self.aspp(m7, acts)# [2,4096,15,13] [ 2,1,120,101]----》[ 2,1536,,15,13]
        dec0_up = self.bot_aspp(x)# [ 2,256,,15,13]

        dec0_fine = self.bot_fine(m2)# [2,48,60,51]
        dec0_up = self.interpolate(dec0_up, m2.size()[2:], mode='bilinear',align_corners=True)## [ 2,256,,60,51]
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)# [ 2,304,,60,51]

        dec1 = self.final_seg(dec0)# [ 2,19,,60,51]
        seg_out = self.interpolate(dec1, x_size[2:], mode='bilinear')    # [ 2,19,120,101]

        return seg_out, edge_out



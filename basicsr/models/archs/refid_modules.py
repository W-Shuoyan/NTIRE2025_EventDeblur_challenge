import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
import torchvision
from torch.nn.modules.utils import _pair, _single
import math

from basicsr.models.archs.fusion_modules import CrossmodalAtten, CrossmodalAtten_imgeventalladd

class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True,
                 max_residue_magnitude=10):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)



class SecondOrderDeformableAlignment(ModulatedDeformConv):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        # return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
        #                                self.stride, self.padding,
        #                                self.dilation, self.groups,
        #                                self.deform_groups)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

############## image encoder #####################
class ImageEncoderConvBlock(nn.Module):
    """
    x conv relu conv relu +  conv_down(k=4 s=2 nobias)
    |------conv-----------|
    """
    def __init__(self, in_size, out_size, downsample, relu_slope): # cat
        super(ImageEncoderConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

        if downsample:
            self.down = conv_down(out_size, out_size, bias=False)


    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)
        if self.downsample:
            out = self.down(out)

        return out
            

class ConvLayer(nn.Module):
    """
    conv norm relu
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu_slope=0.2, norm=None):
        super(ConvLayer, self).__init__()
        self.relu_slope = relu_slope

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if relu_slope is not None:
            if type(relu_slope) is str:
                self.relu = nn.ReLU()
            else:
                self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.relu_slope is not None:
            out = self.relu(out)

        return out



class RecurrentConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x) # downsample first
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state

############### event encoder #########################
class SimpleRecurrentConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_first_dcn=False):
        super(SimpleRecurrentConvLayer, self).__init__()
        self.relu_slope = relu_slope
        # assert(recurrent_block_type in ['convlstm', 'convgru', 'simpleconv'])
        # self.recurrent_block_type = recurrent_block_type
        # print("DEBUG: in recurrent conv layer:{}".format(recurrent_block_type))
        if use_first_dcn:
            self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)

    def forward(self, x, prev_state, bi_direction_state = None):
        x = self.conv(x)
        if self.relu_slope is not None:
            x = self.relu(x)
        x, state = self.recurrent_block(x, prev_state) # tensor
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)

        return x, state


############### new event encoder #########################
class SimpleRecurrentThenDownConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_first_dcn=False):
        super(SimpleRecurrentThenDownConvLayer, self).__init__()
        self.relu_slope = relu_slope
        # assert(recurrent_block_type in ['convlstm', 'convgru', 'simpleconv'])
        # self.recurrent_block_type = recurrent_block_type
        # print("DEBUG: in recurrent conv layer:{}".format(recurrent_block_type))
        if use_first_dcn:
            self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x, prev_state, bi_direction_state = None):
        x = self.conv(x)
        if self.relu_slope is not None:
            x = self.relu(x)
        x, state = self.recurrent_block(x, prev_state) # tensor
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)
        x = self.down(x)
        
        return x, state

############### new event encoder with atten fusion #########################
class SimpleRecurrentThenDownAttenfusionConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_first_dcn=False, use_atten_fuse=False):
        super(SimpleRecurrentThenDownAttenfusionConvLayer, self).__init__()
        self.relu_slope = relu_slope
        self.use_atten_fuse = use_atten_fuse

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_atten_fuse:
            self.atten_fuse = CrossmodalAtten(c=in_channels, c_out = out_channels, DW_Expand=1, FFN_Expand=2)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x, y=None, prev_state=None, bi_direction_state = None):
        # x = self.conv(x)
        # if self.relu_slope is not None:
            # x = self.relu(x)
        if y is not None:
            if self.use_atten_fuse:
                x = self.atten_fuse(x,y)
            else:
                x = x+y
                x = self.conv(x) # increase the c dimension
                if self.relu_slope is not None:
                    x = self.relu(x)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)
        # print('DEBUG: x.shape:{}'.format(x.shape))
        # if prev_state is not None:
            # print('DEBUG: prev_state.shape:{}'.format(prev_state.shape))

        x, state = self.recurrent_block(x, prev_state) # tensor
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)
        x = self.down(x)
        
        return x, state


############### new event encoder with atten fusion and img+event #########################
class SimpleRecurrentThenDownAttenfusionmodifiedConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_first_dcn=False, use_atten_fuse=False):
        super(SimpleRecurrentThenDownAttenfusionmodifiedConvLayer, self).__init__()
        self.relu_slope = relu_slope
        self.use_atten_fuse = use_atten_fuse

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_atten_fuse:
            self.atten_fuse = CrossmodalAtten_imgeventalladd(c=in_channels, c_out = out_channels, DW_Expand=1, FFN_Expand=2)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x, y=None, prev_state=None, bi_direction_state = None):
        # x = self.conv(x)
        # if self.relu_slope is not None:
            # x = self.relu(x)
        if y is not None:
            if self.use_atten_fuse:
                x = self.atten_fuse(x,y)
            else:
                x = x+y
                x = self.conv(x) # increase the c dimension
                if self.relu_slope is not None:
                    x = self.relu(x)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)
        # print('DEBUG: x.shape:{}'.format(x.shape))
        # if prev_state is not None:
            # print('DEBUG: prev_state.shape:{}'.format(prev_state.shape))

        x, state = self.recurrent_block(x, prev_state) # tensor
        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)
        x = self.down(x)
        
        return x, state




class TransposedConvLayer(nn.Module):
    """
    TransConv norm relu
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out



class UpsampleConvLayer(nn.Module):
    """
    bilinear conv norm relu
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposeRecurrentConvLayer(nn.Module):
    """
    for decoder
    transposeconv, recurrent conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, norm=None, fuse_two_direction=False):
        super(TransposeRecurrentConvLayer, self).__init__()
        self.hidden_channel = out_channels
        self.fuse_two_direction = fuse_two_direction
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=True)

        self.forward_trunk = ConvResidualBlocks(out_channels+self.hidden_channel, out_channels, num_block=1)
        if self.fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope=0.2, norm=norm)

    def forward(self, x, prev_state, bi_direction_state=None):
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]
        spatial_size = list(spatial_size)
        for i in range(len(spatial_size)):
            spatial_size[i] *= 2

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_channel] + spatial_size
            prev_state = torch.zeros(state_size).to(x.device)

        out = self.transposed_conv2d(x)
        if self.fuse_two_direction and bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)

        out = torch.cat([out, prev_state], dim=1)
        out = self.forward_trunk(out)
        state = out

        return out, state


class Transpose_simple_ConvLayer(nn.Module):
    """
    for decoder
    transposeconv, recurrent conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, norm=None, fuse_two_direction=False):
        super(Transpose_simple_ConvLayer, self).__init__()
        self.hidden_channel = out_channels
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=True)

        self.forward_trunk = ConvResidualBlocks(out_channels, out_channels, num_block=1)

    def forward(self, x):
        out = self.transposed_conv2d(x)
        out = self.forward_trunk(out)

        return out


class PixelShuffleRecurrentConvLayer(nn.Module):
    """
    for decoder
    transposeconv, recurrent conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, norm=None):
        super(PixelShuffleRecurrentConvLayer, self).__init__()
        self.hidden_channel = out_channels
        self.shuffleupsample = nn.PixelShuffle(2) # pixel shuffle upsample

        self.forward_trunk = ConvResidualBlocks(out_channels+self.hidden_channel, out_channels, num_block=1)

    def forward(self, x, prev_state):
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]
        spatial_size = list(spatial_size)
        for i in range(len(spatial_size)):
            spatial_size[i] *= 2
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_channel] + spatial_size
            prev_state = torch.zeros(state_size).to(x.device)

        out = self.shuffleupsample(x)
        out = torch.cat([out, prev_state], dim=1)
        out = self.forward_trunk(out)
        state = out

        return out, state


class DownsampleRecurrentConvLayer(nn.Module):
    """
    convlstm bilinearDown
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation, 'relu')

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    """
    x conv (norm) relu conv (norm) +x relu
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state



############### new event encoder #########################
class SimpleNoRecurrentThenDownConvLayer(nn.Module):
    """
    conv convlstm
    out_channels = 2* in_channels
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False, use_first_dcn=False):
        super(SimpleNoRecurrentThenDownConvLayer, self).__init__()
        self.relu_slope = relu_slope
        # assert(recurrent_block_type in ['convlstm', 'convgru', 'simpleconv'])
        # self.recurrent_block_type = recurrent_block_type
        # print("DEBUG: in recurrent conv layer:{}".format(recurrent_block_type))
        if use_first_dcn:
            self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.recurrent_block = SimpleNoRecurrentConv(out_channels, 0, num_block=num_block)
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2*out_channels, out_channels, 1, 1, 0, relu_slope, norm)
        self.down = conv_down(out_channels, out_channels, bias=False)

    def forward(self, x):
        x = self.conv(x)
        if self.relu_slope is not None:
            x = self.relu(x)
        x = self.recurrent_block(x) # tensor
        x = self.down(x)
        
        return x


class SimpleRecurrentConv(nn.Module):
    """
    SimpleRecurrentConv, borrowed from BasicVSR
    """

    def __init__(self, input_size, hidden_size, num_block=4):
        super().__init__()
        # propagation
        self.hidden_size = hidden_size
        self.forward_trunk = ConvResidualBlocks(input_size + hidden_size, input_size, num_block)
        # self.fusion = nn.Conv2d(input_size * 2, input_size, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, prev_state):

        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(x.device)

        # backward branch
        feat_prop = torch.cat([x, prev_state], dim=1)
        feat_prop = self.forward_trunk(feat_prop)
        state = feat_prop

        # print("feat_prop type: {}".format(type(feat_prop)))
        # print("state type: {}".format(type(state)))

        return feat_prop, state


class SimpleNoRecurrentConv(nn.Module):
    """
    SimpleNoRecurrentConv
    """

    def __init__(self, input_size, hidden_size=0, num_block=4):
        super().__init__()
        # propagation
        self.hidden_size = hidden_size
        self.forward_trunk = ConvResidualBlocks(input_size + hidden_size, input_size, num_block)
        # self.fusion = nn.Conv2d(input_size * 2, input_size, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        # backward branch
        feat_prop = x
        feat_prop = self.forward_trunk(feat_prop)

        # print("feat_prop type: {}".format(type(feat_prop)))
        # print("state type: {}".format(type(state)))

        return feat_prop





# sub modules
class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)



class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
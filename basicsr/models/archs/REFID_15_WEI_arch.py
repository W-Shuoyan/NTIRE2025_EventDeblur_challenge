import torch
from torch import nn
from einops import rearrange


from basicsr.models.archs.refid_modules import TransposeRecurrentConvLayer, Transpose_simple_ConvLayer, ResidualBlock, ConvLayer, SimpleRecurrentThenDownAttenfusionmodifiedConvLayer, ImageEncoderConvBlock
from basicsr.models.archs.arch_util import make_layer, ResidualBlockNoBN

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2

class FinalDecoderRecurrentUNet(nn.Module):
    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum', activation='sigmoid',
                 num_encoders=3, base_num_channels=32, num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=False):
        super(FinalDecoderRecurrentUNet, self).__init__()

        self.ev_chn = ev_chn
        self.img_chn = img_chn
        self.out_chn = out_chn
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_recurrent_upsample_conv:
            print('Using Recurrent UpsampleConvLayer (slow, but recurrent in decoder)')
            self.UpsampleLayer = TransposeRecurrentConvLayer
        else:
            print('Using No recurrent UpsampleConvLayer (fast, but no recurrent in decoder)')
            self.UpsampleLayer = Transpose_simple_ConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.ev_chn > 0)
        assert(self.img_chn > 0)
        assert(self.out_chn > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_indexs = []
        for i in range(self.num_encoders):
            self.encoder_indexs.append(i)

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks_forward = nn.ModuleList()
        self.resblocks_backward = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks_forward.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))
            self.resblocks_backward.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=2, padding=0, norm=self.norm)) # kernei_size= 5, padidng =2 before
            self.fusions.append(ConvLayer(2*input_size, input_size, 1, 1, 0, relu_slope=0.2))

    def build_prediction_layer(self):
        self.enhance_tail = make_layer(ResidualBlockNoBN, 5, num_feat=self.base_num_channels)
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.out_chn, kernel_size=3, stride=1, padding=1, relu_slope=None, norm=self.norm)



class FinalBidirectionAttenfusion_WEI(FinalDecoderRecurrentUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.

    num_block: the number of blocks in each simpleconvlayer.
    """

    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=False, num_block=3, use_first_dcn=False, use_reversed_voxel=False):
        super(FinalBidirectionAttenfusion_WEI, self).__init__(img_chn, ev_chn, out_chn, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_recurrent_upsample_conv)
        self.use_reversed_voxel = use_reversed_voxel
        ## event
        self.head = ConvLayer(self.ev_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()

        for input_size, output_size, encoder_index in zip(self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs):
            # print('DEBUG: input size:{}'.format(input_size))
            # print('DEBUG: output size:{}'.format(output_size))
            print('Using enhanced attention!')
            use_atten_fuse = True if encoder_index == 1 else False
            self.encoders_backward.append(SimpleRecurrentThenDownAttenfusionmodifiedConvLayer(input_size, output_size,
                                                    kernel_size=3, stride=1, padding=1, fuse_two_direction=False,
                                                    norm=self.norm, num_block=num_block, use_first_dcn=use_first_dcn,
                                                    use_atten_fuse=use_atten_fuse))
                                                    
            self.encoders_forward.append(SimpleRecurrentThenDownAttenfusionmodifiedConvLayer(input_size, output_size,
                                                    kernel_size=3, stride=1, padding=1, fuse_two_direction=False,
                                                    norm=self.norm, num_block=num_block, use_first_dcn=use_first_dcn,
                                                    use_atten_fuse=use_atten_fuse))

        ## img
        self.head_img = ConvLayer(self.img_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.img_encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.img_encoders.append(ImageEncoderConvBlock(in_size=input_size, out_size=output_size,
                                                            downsample=True, relu_slope=0.2))
        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, image, event):
        """
        :param x: b 2 c h w -> b, 2c, h, w
        :param event: b, t, num_bins, h, w -> b*t num_bins(2) h w 
        :return: b, t, out_chn, h, w

        One direction propt version
        TODO:  use_reversed_voxel!!!
        """
        # reshape
        x = image
        if x.dim()==5:
            x = rearrange(x, 'b t c h w -> b (t c) h w') # sharp
        b, t, num_bins, h, w = event.size()
        event = rearrange(event, 'b t c h w -> (b t) c h w')

        
        # head
        x = self.head_img(x) # image feat
        head = x
        e = self.head(event)   # event feat
        # image encoder
        x_blocks = []
        for i, img_encoder in enumerate(self.img_encoders):
            x = img_encoder(x)
            x_blocks.append(x)

########
        ## prepare for propt 
        e = rearrange(e, '(b t) c h w -> b t c h w', b=b, t=t)

        center_idx = (t - 1) // 2
        backward_prev_states = [None] * self.num_encoders # prev states for each scale
        backward_e_blocks = [None] * self.num_encoders  
        ## backward propt
        for frame_idx in range(t-1, center_idx-1, -1):
            e_cur = e[:, frame_idx,:,:,:] # b,c,h,w
            for i, back_encoder in enumerate(self.encoders_backward):
                if i==0:
                    e_cur, state = back_encoder(x=e_cur,y=None, prev_state=backward_prev_states[i])
                else:
                    e_cur, state = back_encoder(x=e_cur,y=x_blocks[i-1], prev_state=backward_prev_states[i])
                backward_e_blocks[i] = e_cur
                backward_prev_states[i] = state
        # residual blocks
        for i in range(len(self.resblocks_backward)):
            if i == 0:
                e_cur = self.resblocks_backward[i](e_cur+x_blocks[-1])
            else:
                e_cur = self.resblocks_backward[i](e_cur)
        backward_e_blocks[-1] = e_cur
        
        forward_prev_states = [None] * self.num_encoders # prev states for each scale
        forward_e_blocks = [None] * self.num_encoders 
        ## forward propt 
        for frame_idx in range(0,center_idx+1):
            e_cur = e[:, frame_idx,:,:,:] # b,c,h,w
            for i, encoder in enumerate(self.encoders_forward):
                if i==0:
                    e_cur, state = encoder(x=e_cur, y=None, prev_state=forward_prev_states[i])
                else:
                    e_cur, state = encoder(x=e_cur, y=x_blocks[i-1], prev_state=forward_prev_states[i])
                forward_e_blocks[i] = e_cur
                forward_prev_states[i] = state # update state for next frame
        # residual blocks
        for i in range(len(self.resblocks_forward)):
            if i == 0:
                e_cur = self.resblocks_forward[i](e_cur+x_blocks[-1])
            else:
                e_cur = self.resblocks_forward[i](e_cur)
        forward_e_blocks[-1] = e_cur

##########################################################################
        ## Decoder
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                e_cur = self.fusions[i](torch.cat((backward_e_blocks[self.num_encoders - i - 1], forward_e_blocks[self.num_encoders - i - 1]), 1))
                e_cur = decoder(e_cur)
            else:
                e_skip = self.fusions[i](torch.cat((backward_e_blocks[self.num_encoders - i - 1], forward_e_blocks[self.num_encoders - i - 1]), 1))
                e_cur = decoder(self.apply_skip_connection(e_cur, e_skip))

        # tail
        out = self.enhance_tail(e_cur) + head
        out = self.pred(out)
        return out
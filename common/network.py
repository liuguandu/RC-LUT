import torch
import torch.nn as nn
import torch.nn.functional as F


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))
def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)
class Conv_test(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv_test, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, in_channels, 1,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        if bias:
            nn.init.constant_(self.conv1.bias, 0)
            nn.init.constant_(self.conv2.bias, 0)
            nn.init.constant_(self.conv3.bias, 0)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class RC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, out_dim, kernel_size=4, stride=1, padding=0, dilation=1, bias=True, mlp_field=7):
        super(RC_Module, self).__init__()
        self.mlp_field = mlp_field
        self.conv = nn.Conv2d(in_channels, out_channels, 1,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.out_layer = nn.Linear(out_channels, in_channels)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        for i in range(mlp_field * mlp_field):
            setattr(self, 'linear{}'.format(i+1), nn.Linear(in_channels, out_channels))
            setattr(self, 'out{}'.format(i+1), nn.Linear(out_channels, out_dim))
        
    def forward(self, x):
        x_kv = {}
        # print('x', x.size())
        for i in range(self.mlp_field):
            for j in range(self.mlp_field):
                num = i * self.mlp_field + j + 1
                module1 = getattr(self, 'linear{}'.format(num))
                x_kv[str(num)] = module1(x[:, i, j, :]).unsqueeze(1)
        x_list = []
        temp = []
        for i in range(self.mlp_field * self.mlp_field):
            module = getattr(self, 'out{}'.format(i+1))
            x_list.append(module(x_kv[str(i+1)]))
        # for i in range(self.mlp_field * self.mlp_field):
        #     temp.append(x_kv[str(i+1)])
        out = torch.cat(x_list, dim=1)
        # out = torch.cat(temp, dim=1)
        out = out.mean(1)
        # out = self.out_layer(out)
        out = out.unsqueeze(-1).unsqueeze(-1)
        
        out = torch.tanh(out)
        out = round_func(out * 127)
        bias, norm = 127, 255.0
        out = round_func(torch.clamp(out + bias, 0, 255)) / norm
        return out

class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


############### MuLUT Blocks ###############
class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, upscale=1, out_c=1, dense=True, deform_mode='s', patch_size=48, stage=1):
        super(MuLUTUnit, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale
        self.conv_naive = Conv(1, nf, 2)
        self.mode = mode
        self.stage = stage

        if mode == '2x2':
            
            if self.stage == 1:
                self.conv1 = RC_Module(1, nf, 4, mlp_field=5)
            else:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=5)
            self.s_conv = Conv(4, nf, 1)
            # self.conv1 = Conv_test(1, nf)
        elif mode == '2x2d':
            # self.conv1 = Conv(1, nf, 2, dilation=2)
            if self.stage == 1:
                self.conv1 = RC_Module(1, nf, 4, mlp_field=7)
            else:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=7)
            self.d_conv = Conv(4, nf, 1)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '2x2d4':
            self.conv1 = Conv(1, nf, 2, dilation=4)
        elif mode == '1x4':
            # self.conv1 = Conv(1, nf, (1, 4))
            if self.stage == 1:
                self.conv1 = RC_Module(1, nf, 4, mlp_field=3)
            else:
                self.conv1 = RC_Module(1, nf, 1, mlp_field=3)
            self.y_conv = Conv(4, nf, 1)
        elif mode == '4x1':
            self.conv1 = DeformConv2d(1, nf, mode=deform_mode)
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)
            # self.conv6 = Conv(nf * 5, nf, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x, r_H, r_W, x_dense, x_3x3, x_7x7):
        B, C, H, W = x_dense.shape
        x_dense = x_dense.reshape(-1, 1, H, W)
        if self.mode == '2x2':
            x = x
            x = torch.tanh(self.conv1(x))
            if self.stage == -1:
                x = self.s_conv(x)
            else:
                x = x.reshape(-1, 1, H, W)
                # x += x_dense
                x = F.pad(x, [0, 1, 0, 1], mode='replicate')
                x = F.unfold(x, 2)
                x = x.view(B, C, 2*2, H*W)
                x = x.permute((0,1,3,2))
                x = x.reshape(B * C * H * W, 2, 2)
                x = x.unsqueeze(1)
                x = self.act(self.conv_naive(x))
            
        elif self.mode == '2x2d':
            x = x_7x7
            x = torch.tanh(self.conv1(x))
            if self.stage == -1:
                x = self.d_conv(x)
            else:
                x = x.reshape(-1, 1, H, W)
                # x += x_dense
                x = F.pad(x, [0, 1, 0, 1], mode='replicate')
                x = F.unfold(x, 2)
                x = x.view(B, C, 2*2, H*W)
                x = x.permute((0,1,3,2))
                x = x.reshape(B * C * H * W, 2, 2)
                x = x.unsqueeze(1)
                x = self.act(self.conv_naive(x))
        elif self.mode == '1x4':
            x = x_3x3
            x = torch.tanh(self.conv1(x))
            if self.stage == -1:
                x = self.y_conv(x)
            else:
                x = x.reshape(-1, 1, H, W)
                # x += x_dense
                x = F.pad(x, [0, 1, 0, 1], mode='replicate')
                x = F.unfold(x, 2)
                x = x.view(B, C, 2*2, H*W)
                x = x.permute((0,1,3,2))
                x = x.reshape(B * C * H * W, 2, 2)
                x = x.unsqueeze(1)
                x = self.act(self.conv_naive(x))

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x += x_3x3
        x = torch.tanh(x)
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        # print(x.size())
        return x


## cheap 1 conv 3->1, 
class MuLUTcUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, mode, nf):
        super(MuLUTcUnit, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(3, 3, 1)
        self.conv2 = Conv(3, 1, 1)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=0, stride=1, modulation=False, mode='s'):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.mode = mode
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size)
        # self.p_conv1 = nn.Conv2d(inc, outc, kernel_size=2, padding=padding, stride=stride)
        # self.p_conv2 = nn.Conv2d(outc, outc, kernel_size=1, padding=padding, stride=stride)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=padding, stride=stride)
        self.p_conv_d = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=2, padding=padding, stride=stride, dilation=2)
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv_d.weight, 0)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=2, padding=padding, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, self.kernel_size),
            torch.arange(0, self.kernel_size)
        ) # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    def _get_p_n_dilation(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.tensor([0, 2]),
            torch.tensor([0, 2])
        ) # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    def _get_p_n_y(self, N, dtype):
        # p_n_x, p_n_y = torch.meshgrid(
        #     torch.tensor([0, 2]),
        #     torch.tensor([0, 2])
        # ) # (2N, 1)
        # p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = torch.tensor([0, 1, 1, 2, 0, 1, 2, 1])
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n
    def _get_p_0(self, h, w, N, dtype):
        # print('h', h, 'w', w, 'N', N)
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h*self.stride, self.stride),
            torch.arange(0, w*self.stride, self.stride)
        )
        # print('p_0_x', p_0_x.size(), 'p_0_y', p_0_y.size())
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype, mode):
        # print('offset', offset.size())
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        if mode == 's' or mode == 'h':
            p_n = self._get_p_n(N, dtype)
            # print('s', p_n)
        elif mode == 'd':
            p_n = self._get_p_n_dilation(N, dtype)
            # print('d', p_n)
        elif mode == 'y':
            p_n = _get_p_n_y(N, dtype)
            # print('y', p_n)
        else:
            raise AttributeError
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        # print('p_0', p_0.size(), 'p_n', p_n.size(), 'p', p.size())
        # p_mid = p_0.repeat(p.size(0), 1, 1, 1)
        # print('p_0', p_0.size(), 'p_n', p_n.size(), 'p', p.size(), 'p_mid', p_mid.size())
        # p[:, 0, :, :] = p_mid[:, 0, :, :]
        # p[:, N, :, :] = p_mid[:, N, :, :]
        return p
    def _get_p_origin(self, offset, dtype, mode):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        if mode == 's':
            p_n = self._get_p_n(N, dtype)
            # print('s', p_n)
        elif mode == 'd':
            p_n = self._get_p_n_dilation(N, dtype)
            # print('d', p_n)
        elif mode == 'y':
            p_n = _get_p_n_y(N, dtype)
            # print('y', p_n)
        else:
            raise AttributeError
        
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n
        # print('p_n', p_n, 'p', p)
        # print('p_0', p_0.size(), 'p_n', p_n.size(), 'p', p.size())
        # p = p.floor()
        # print('p', p)
        return p
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
    def forward(self, x):
        # offset = self.p_conv(self.p_conv2(self.p_conv1(x)))
        offset = self.p_conv(x)
        if self.mode == 's':
            offset = self.p_conv(x)
        elif self.mode == 'd':
            offset = self.p_conv_d(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1)//2
        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype, self.mode)
        # p_origin = self._get_p_origin(offset, dtype, self.mode)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # p_origin = p_origin.contiguous().permute(0, 2, 3, 1)
        # p_origin = p_origin.detach().floor().long()
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # x_origin = self._get_x_q(x, p_origin, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        
        # x_origin = self._reshape_x_offset(x_origin, ks)
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        # out = self.conv(x)
        # out += x[:, :, :-1, :-1]
        return out

############### Image Super-Resolution ###############
class SRNet(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block. 
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, upscale=None, dense=True):
        super(SRNet, self).__init__()
        self.mode = mode
        
        if 'x1' in mode:
            assert upscale is None
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, upscale=1, dense=dense, stage=1)
            # self.model = MuLUTUnit('4x1', nf, upscale=1, dense=dense, deform_mode='s')
            self.K = 5
            self.S = 1
        elif mode == 'SxN':
            upscale = upscale
            self.model = MuLUTUnit('2x2', nf, upscale=upscale, dense=dense, stage=2)
            # self.model = MuLUTUnit('4x1', nf, upscale=upscale, dense=dense, deform_mode='s')
            self.K = 5
            self.S = upscale
        elif mode == 'Hx1':
            self.model = MuLUTUnit('4x1', nf, upscale=1, dense=dense, deform_mode='h')
            self.K = 2
            self.S = 1
        elif mode == 'HxN':
            self.model = MuLUTUnit('4x1', nf, upscale=upscale, dense=dense, deform_mode='h')
            self.K = 2
            self.S = upscale
        elif mode == 'Jx1':
            self.model = MuLUTUnit('4x1', nf, upscale=1, dense=dense, deform_mode='h')
            self.K = 2
            self.S = 1
        elif mode == 'JxN':
            self.model = MuLUTUnit('4x1', nf, upscale=upscale, dense=dense, deform_mode='h')
            self.K = 2
            self.S = upscale
        elif mode == 'Fx1':
            self.model = MuLUTUnit('2x2d4', nf, upscale=1, dense=dense)
            self.K = 5
            self.S = 1
        elif mode == 'FxN':
            self.model = MuLUTUnit('2x2d4', nf, upscale=upscale, dense=dense)
            self.K = 5
            self.S = upscale
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, upscale=1, dense=dense, stage=1)
            # self.model = MuLUTUnit('4x1', nf, upscale=1, dense=dense, deform_mode='d')
            self.K = 5
            self.S = 1
        elif mode == 'DxN':
            self.model = MuLUTUnit('2x2d', nf, upscale=upscale, dense=dense, stage=2)
            # self.model = MuLUTUnit('4x1', nf, upscale=upscale, dense=dense, deform_mode='d')
            self.K = 5
            self.S = upscale
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, upscale=1, dense=dense, stage=1)
            # self.model = MuLUTUnit('4x1', nf, upscale=1, dense=dense, deform_mode='y')
            self.K = 5
            self.S = 1
        elif mode == 'YxN':
            self.model = MuLUTUnit('1x4', nf, upscale=upscale, dense=dense, stage=2)
            # self.model = MuLUTUnit('4x1', nf, upscale=upscale, dense=dense, deform_mode='y')
            self.K = 5
            self.S = upscale
        elif mode == 'Ex1':
            self.model = MuLUTUnit('2x2d3', nf, upscale=1, dense=dense)
            self.K = 4
            self.S = 1
        elif mode == 'ExN':
            self.model = MuLUTUnit('2x2d3', nf, upscale=upscale, dense=dense)
            self.K = 4
            self.S = upscale
        # elif mode in ['Ox1', 'Hx1']:
        #     self.model = MuLUTUnit('1x4', nf, upscale=1, dense=dense)
        #     self.K = 4
        #     self.S = 1
        # elif mode == ['OxN', 'HxN']:
        #     self.model = MuLUTUnit('1x4', nf, upscale=upscale, dense=dense)
        #     self.K = 4
        #     self.S = upscale
        elif mode == 'Connect':
            self.model = MuLUTcUnit('1x1', nf)
            self.K = 3
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        if 'H' in self.mode:
            channel = x.size(1)
            x = x.reshape(-1, 1, x.size(2), x.size(3))
            x = self.model(x)
            x = x.reshape(-1, channel, x.size(2), x.size(3))
        elif self.mode == 'Connect':
            x = self.model(x)
        else:
            B, C, H, W = x.shape
            x_dense = x[:, :, :-4, :-4]
            x_7x7 = F.pad(x, [2, 0, 2, 0], mode='replicate')
            B7, C7, H7, W7 = x_7x7.shape
            x_7x7 = F.unfold(x_7x7, 7) 
            x_3x3 = x[:, :, :-2, :-2]
            B3, C3, H3, W3 = x_3x3.shape
            x_3x3 = F.unfold(x_3x3, 3)
            
            x_3x3 = x_3x3.view(B3, C3, 9, (H3-2)*(W3-2))
            x_3x3 = x_3x3.permute((0, 1, 3, 2))
            x_3x3 = x_3x3.reshape(B3 * C3 * (H3-2)*(W3-2), 3, 3)
            x_3x3 = x_3x3.unsqueeze(-1)

            x_7x7 = x_7x7.view(B7, C7, 49, (H7-6)*(W7-6))
            x_7x7 = x_7x7.permute((0, 1, 3, 2))
            x_7x7 = x_7x7.reshape(B7 * C7 * (H7-6)*(W7-6), 7, 7)
            x_7x7 = x_7x7.unsqueeze(-1)

            x = F.unfold(x, self.K)  # B,C*K*K,L
            x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
            r_H = H - self.P
            r_W = W - self.P
            x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
            x = x.reshape(B * C * (H - self.P) * (W - self.P),
                        self.K, self.K)  # B*C*L,K,K
            # x = x.unsqueeze(1)  # B*C*L,l,K,K
            x = x.unsqueeze(-1)

            # if 'Y' in self.mode:
            #     x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
            #                 x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            #     x = x.unsqueeze(1).unsqueeze(1)
            if 'H' in self.mode:
                x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                            x[:, :, 2, 3], x[:, :, 3, 2]], dim=1)

                x = x.unsqueeze(1).unsqueeze(1)
            elif 'O' in self.mode:
                x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                            x[:, :, 1, 3], x[:, :, 3, 1]], dim=1)

                x = x.unsqueeze(1).unsqueeze(1)

            x = self.model(x, r_H, r_W, x_dense, x_3x3, x_7x7)   # B*C*L,K,K
            x = x.squeeze(1)
            x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
            x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
            x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
            x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                    self.S, stride=self.S)
            # print('ll', x.size())
        return x


############### Grayscale Denoising, Deblocking, Color Image Denosing ###############
class DNNet(nn.Module):
    """ Wrapper of basic MuLUT block without upsampling. """

    def __init__(self, mode, nf=64, dense=True):
        super(DNNet, self).__init__()
        self.mode = mode

        self.S = 1
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, dense=dense)
            self.K = 2
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, dense=dense)
            self.K = 3
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, dense=dense)
            self.K = 3
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)   # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))     # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x


############### Image Demosaicking ###############
class DMNet(nn.Module):
    """ Wrapper of the first stage of MuLUT network for demosaicking. 4D(RGGB) bayer patter to (4*3)RGB"""

    def __init__(self, mode, nf=64, dense=False):
        super(DMNet, self).__init__()
        self.mode = mode

        if mode == 'SxN':
            self.model = MuLUTUnit('2x2', nf, upscale=2, out_c=3, dense=dense)
            self.K = 2
            self.C = 3
        else:
            raise AttributeError
        self.P = 0  # no need to add padding self.K - 1
        self.S = 2  # upscale=2, stride=2

    def forward(self, x):
        B, C, H, W = x.shape
        # bayer pattern, stride = 2
        x = F.unfold(x, self.K, stride=2)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H // 2) * (W // 2))  # stride = 2
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H // 2) * (W // 2),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        # print("in", torch.round(x[0, 0]*255))

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,out_C,S,S
        # self.C along with feat scale
        x = x.reshape(B, C, (H // 2) * (W // 2), -1)  # B,C,L,out_C*S*S
        x = x.permute((0, 1, 3, 2))  # B,C,outC*S*S,L
        x = x.reshape(B, -1, (H // 2) * (W // 2))  # B,C*out_C*S*S,L
        x = F.fold(x, ((H // 2) * self.S, (W // 2) * self.S),
                   self.S, stride=self.S)
        return x

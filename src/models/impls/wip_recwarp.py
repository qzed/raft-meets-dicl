import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, Result


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class GaDeconv2xBlock(nn.Module):
    """Deconvolution + convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(c_in, c_out, bias=False, kernel_size=4, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class FeatureNet(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_channels):
        super().__init__()

        self.conv0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1),
            ConvBlock(32, 32, kernel_size=3, padding=1, stride=2),
            ConvBlock(32, 32, kernel_size=3, padding=1),
        )

        self.conv1a = ConvBlock(32, 48, kernel_size=3, padding=1, stride=2)
        self.conv2a = ConvBlock(48, 64, kernel_size=3, padding=1, stride=2)
        self.conv3a = ConvBlock(64, 96, kernel_size=3, padding=1, stride=2)
        self.conv4a = ConvBlock(96, 128, kernel_size=3, padding=1, stride=2)
        self.conv5a = ConvBlock(128, 160, kernel_size=3, padding=1, stride=2)
        self.conv6a = ConvBlock(160, 192, kernel_size=3, padding=1, stride=2)

        self.deconv6a = GaDeconv2xBlock(192, 160)
        self.deconv5a = GaDeconv2xBlock(160, 128)
        self.deconv4a = GaDeconv2xBlock(128, 96)
        self.deconv3a = GaDeconv2xBlock(96, 64)
        self.deconv2a = GaDeconv2xBlock(64, 48)
        self.deconv1a = GaDeconv2xBlock(48, 32)

        self.conv1b = GaConv2xBlock(32, 48)
        self.conv2b = GaConv2xBlock(48, 64)
        self.conv3b = GaConv2xBlock(64, 96)
        self.conv4b = GaConv2xBlock(96, 128)
        self.conv5b = GaConv2xBlock(128, 160)
        self.conv6b = GaConv2xBlock(160, 192)

        self.deconv6b = GaDeconv2xBlock(192, 160)
        self.outconv6 = ConvBlock(160, output_channels, kernel_size=3, padding=1)

        self.deconv5b = GaDeconv2xBlock(160, 128)
        self.outconv5 = ConvBlock(128, output_channels, kernel_size=3, padding=1)

        self.deconv4b = GaDeconv2xBlock(128, 96)
        self.outconv4 = ConvBlock(96, output_channels, kernel_size=3, padding=1)

        self.deconv3b = GaDeconv2xBlock(96, 64)
        self.outconv3 = ConvBlock(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = res0 = self.conv0(x)                # -> 32, H/2, W/2

        x = res1 = self.conv1a(x)               # -> 48, H/4, W/4
        x = res2 = self.conv2a(x)               # -> 64, H/8, W/8
        x = res3 = self.conv3a(x)               # -> 96, H/16, W/16
        x = res4 = self.conv4a(x)               # -> 128, H/32, W/32
        x = res5 = self.conv5a(x)               # -> 160, H/64, W/64
        x = res6 = self.conv6a(x)               # -> 192, H/128, W/128

        x = res5 = self.deconv6a(x, res5)       # -> 160, H/64, W/64
        x = res4 = self.deconv5a(x, res4)       # -> 128, H/32, W/32
        x = res3 = self.deconv4a(x, res3)       # -> 96, H/16, W/16
        x = res2 = self.deconv3a(x, res2)       # -> 64, H/8, W/8
        x = res1 = self.deconv2a(x, res1)       # -> 48, H/4, W/4
        x = res0 = self.deconv1a(x, res0)       # -> 32, H/2, W/2

        x = res1 = self.conv1b(x, res1)         # -> 48, H/4, W/4
        x = res2 = self.conv2b(x, res2)         # -> 64, H/8, W/8
        x = res3 = self.conv3b(x, res3)         # -> 96, H/16, W/16
        x = res4 = self.conv4b(x, res4)         # -> 128, H/32, W/32
        x = res5 = self.conv5b(x, res5)         # -> 160, H/64, W/64
        x = res6 = self.conv6b(x, res6)         # -> 192, H/128, W/128

        x = self.deconv6b(x, res5)              # -> 160, H/64, W/64
        x6 = self.outconv6(x)                   # -> 32, H/64, W/64

        x = self.deconv5b(x, res4)              # -> 128, H/32, W/32
        x5 = self.outconv5(x)                   # -> 32, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 32, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 32, H/8, W/8

        return x3, x4, x5, x6


class MatchingNet(nn.Sequential):
    def __init__(self, feature_channels):
        super().__init__(
            ConvBlock(2 * feature_channels, 96, kernel_size=3, padding=1),
            ConvBlock(96, 128, kernel_size=3, padding=1, stride=2),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            ConvBlock(128, 64, kernel_size=3, padding=1),
            DeconvBlock(64, 32, kernel_size=4, padding=1, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),     # note: with bias
        )

    def forward(self, x):
        batch, v, u, c, h, w = x.shape

        x = x.view(batch * v * u, c, h, w)
        x = super().forward(x)
        x = x.view(batch, v, u, h, w)

        return x


class FlowRegression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost):
        batch, v, u, h, w = cost.shape
        du, dv = (u - 1) // 2, (v - 1) // 2             # displacement range

        # displacement offsets along u
        disp_u = torch.arange(-du, du + 1, device=cost.device, dtype=torch.float32)
        disp_u = disp_u.view(1, u)
        disp_u = disp_u.expand(v, -1)                   # expand for stacking

        # displacement offsets along v
        disp_v = torch.arange(-dv, dv + 1, device=cost.device, dtype=torch.float32)
        disp_v = disp_v.view(v, 1)
        disp_v = disp_v.expand(-1, u)                   # expand for stacking

        # combined displacement vector
        disp = torch.stack((disp_u, disp_v), dim=0)     # stack coordinates to (2, v, u)
        disp = disp.view(1, 2, v, u, 1, 1)              # reshape for broadcasting

        # compute displacement probability
        cost = cost.view(batch, v * u, h, w)            # combine disp. dimensions for softmax
        prob = F.softmax(cost, dim=1)                   # softmax along displacement dimensions
        prob = prob.view(batch, 1, v, u, h, w)          # reshape for broadcasting

        # compute flow as weighted sum over displacement vectors
        flow = (prob * disp).sum(dim=(2, 3))            # weighted sum over displacements

        return flow                                     # (batch, 2, h, w)


class RecurrentFlowUnit(nn.Module):
    def __init__(self, feature_channels, range):
        super().__init__()

        self.disp = range

        self.mnet = MatchingNet(feature_channels)
        self.flow = FlowRegression()

    def forward(self, feat1, feat2, coords):
        # collect second frame features: warp backwards with displacement context
        feat2 = self.warp_with_context(feat2, coords, self.disp)    # (batch, 2dv+1, 2du+1, c, h, w)

        # concatenate first and second frame features along channels
        feat = self.feature_stack(feat1, feat2)                 # (batch, 2dv+1, 2du+1, 2c, h, w)

        # compute matching costs/correlation from features
        cost = self.mnet(feat)                                  # (batch, 2dv+1, 2du+1, h, w)

        # flow delta regression
        delta = self.flow(cost)                                 # (batch, 2, h, w)

        # update current flow coordinates
        return coords + delta

    def lookup_kernel(self, shape, device):
        du, dv = shape

        cx = torch.linspace(-du, du, 2 * du + 1, dtype=torch.float, device=device)
        cy = torch.linspace(-dv, dv, 2 * dv + 1, dtype=torch.float, device=device)

        delta = torch.meshgrid(cy, cx)[::-1]
        delta = torch.stack(delta, dim=-1)

        return delta                                            # (2dv+1, 2du+1, 2)

    def warp_with_context(self, feat2, coords, disp):
        batch, c, h, w = feat2.shape
        du, dv = disp

        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, h, w, 1, 1, 2)

        # add lookup kernel
        coords = coords + self.lookup_kernel((du, dv), device=coords.device)

        # normalize coordinates for sampling
        coords[..., 0] = 2 * coords[..., 0] / (w - 1) - 1
        coords[..., 1] = 2 * coords[..., 1] / (h - 1) - 1

        # prepare for sampling
        coords = coords.reshape(batch, h * w, 2*dv + 1, 2*du + 1, 2)

        # sample second frame: backwards warping with context
        # FIXME: possible to build a custom grid_sample that does not run into these memory issues?
        f2o = []
        for b in range(batch):
            f2b = feat2[b]

            f2b = f2b.view(1, c, h, w)
            f2b = f2b.expand(h*w, c, h, w)

            f2b = F.grid_sample(f2b, coords[b], align_corners=True)

            f2b = f2b.view(h, w, c, 2*dv + 1, 2*du + 1)
            f2b = f2b.permute(3, 4, 2, 0, 1)                    # (2dv+1, 2du+1, c, h, w)

            f2o.append(f2b)

        return torch.stack(f2o, dim=0)                          # (batch, 2dv+1, 2du+1, c, h, w)

    def feature_stack(self, feat1, feat2):
        batch, v, u, c, h, w = feat2.shape

        # prepare first frame for concatenation
        feat1 = feat1.view(batch, 1, 1, c, h, w)
        feat1 = feat1.expand(-1, v, u, -1, -1, -1)              # (batch, 2dv+1, 2du+1, c, h, w)

        # concatenate first and second frame features along channels
        return torch.cat((feat1, feat2), dim=-3)                # (batch, 2dv+1, 2du+1, 2c, h, w)


class WipModule(nn.Module):
    def __init__(self, feature_channels=32, disp=[(3, 3)]*4):
        super().__init__()

        self.fnet = FeatureNet(feature_channels)
        self.rfu = nn.ModuleList([RecurrentFlowUnit(feature_channels, disp[i]) for i in range(4)])

    def forward(self, img1, img2, iterations=[1, 1, 1, 1, 1]):
        batch, _, h, w = img1.shape

        # perform feature extraction
        feat1 = self.fnet(img1)
        feat2 = self.fnet(img2)

        # initialize flow
        coords = self.coordinate_grid((batch, *feat1[-1].shape[2:]), device=img1.device)

        # coarse-to-fine warping
        out = []
        for i, (f1, f2) in reversed(list(enumerate(zip(feat1, feat2)))):
            _, _, h2, w2 = f1.shape

            # upsample
            if coords.shape[2:] != f1.shape[2:]:
                _, _, h1, w1 = coords.shape

                coords = F.interpolate(coords, (h2, w2), mode='bilinear', align_corners=True)
                coords[:, 0, ...] = coords[:, 0, ...] * (w2 / w1)
                coords[:, 1, ...] = coords[:, 1, ...] * (h2 / h1)

            # recurrently update coordinates
            coords0 = self.coordinate_grid((batch, *f1.shape[2:]), device=img1.device)

            for j in range(iterations[i]):
                coords = self.rfu[i](f1, f2, coords)

                out.append(coords - coords0)

        return out

    def coordinate_grid(self, shape, device):
        batch, h, w = shape

        cy = torch.arange(h, dtype=torch.float, device=device)
        cx = torch.arange(w, dtype=torch.float, device=device)

        coords = torch.meshgrid(cy, cx)[::-1]               # build transposed grid (h, w) x 2
        coords = torch.stack(coords, dim=0)                 # combine coordinates (2, h, w)
        coords = coords.expand(batch, -1, -1, -1)           # expand to batch (batch, 2, h, w)

        return coords


class Wip(Model):
    type = 'wip/warp/2'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        feat = param_cfg.get('feature-channels', 32)
        disp = param_cfg.get('disp-range', [(3, 3)] * 4)

        args = cfg.get('arguments', {})

        return cls(feat, disp, args)

    def __init__(self, feature_channels, disp, arguments={}):
        super().__init__(WipModule(feature_channels, disp), arguments)

        self.feature_channels = feature_channels
        self.disp = disp

    def get_config(self):
        default_args = {'iterations': [1]*4}

        return {
            'type': self.type,
            'parameters': {
                'feature-channels': self.feature_channels,
                'range': self.disp,
            },
            'arguments': default_args | self.arguments,
        }

    def forward(self, img1, img2, iterations=[1]*4):
        _, _, h, w = img1.shape
        return WipResult(self.module(img1, img2, iterations), img1.shape[2:])


class WipResult(Result):
    def __init__(self, output, shape):
        super().__init__()

        self.result = list(reversed(output))
        self.shape = shape

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [x[batch_index].view(1, *x.shape[1:]) for x in self.result]

    def final(self):
        flow = self.result[0]

        _b, _c, fh, fw = flow.shape
        th, tw = self.shape

        flow = F.interpolate(flow.detach(), (th, tw), mode='bilinear', align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow

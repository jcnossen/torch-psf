"""SMLMNet: the detection network of SIMCODE v1 in its plain-SMLM form (a DECODE-style network, Speiser et al. 2021).

A window of T consecutive frames (photoelectrons / px; T = 3: frame t and its two neighbours) -> per-pixel maps of
frame t: detection probability p, the means mu and standard deviations sigma of x, y (px, image coordinates: pixel
centres at integers, x = column), z (um), bg (photoelectrons / px in frame t) and photons (photoelectrons of the
emitter in frame t), and with cov='full_xyz' three partial correlations of x, y, z.

Architecture: a frame U-Net with shared weights runs on every frame of the window (1 -> frame_features[0] channels),
the T feature maps are concatenated and a window U-Net (window_features) maps them to window_out channels, then a head
of grouped 1 x 1 convolutions (one group per output channel) gives the raw outputs; with cov='full_xyz' a separate
1 x 1 convolution (zero-initialised) gives the three correlation channels. Every convolution has 'same' padding, so
the frame size must be a multiple of 2^(len(window_features) - 1) (4 for the default) and a multiple of
2^(len(frame_features) - 1).

Output [B, C, H, W] with C = 1 + 2 n (+ 3 for full_xyz), n = 5 (x, y, z, bg, photons) or 4 without z:
  channel 0           p = sigmoid
  channels 1 .. n     mu = tanh x scale, plus the pixel coordinate for x and y (offsets of at most xyz_scale px)
  channels n+1 .. 2n  sigma = (sigmoid clipped at 0) + 1e-3, x scale (no near-zero sigmas: DECODE)
  channels 2n+1 ..    rho_xy, rho_xz, rho_yz|x = 0.95 tanh (partial correlations; covariance: xyz_cholesky)
scale = xyz_scale (px, px, um), max_bg, max_photons. The input is (frames - input_offset) x input_scale.
"""
import numpy as np
import torch
import torch.nn as nn

PARAMS_3D = ('x', 'y', 'z', 'bg', 'photons')
PARAMS_2D = ('x', 'y', 'bg', 'photons')


class Conv(nn.Module):
    """Conv2d ('same' padding), optional BatchNorm2d, activation."""
    def __init__(self, cin, cout, activation, batch_norm=False, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=kernel_size, padding='same')
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)


class Down(nn.Module):
    """Two Conv blocks, then a stride-2 2 x 2 convolution (downsampling). forward -> (skip, downsampled)."""
    def __init__(self, cin, cout, activation, batch_norm):
        super().__init__()
        self.convs = nn.Sequential(Conv(cin, cout, activation, batch_norm), Conv(cout, cout, activation, batch_norm))
        self.pool = nn.Conv2d(cout, cout, kernel_size=2, stride=2)

    def forward(self, x):
        s = self.convs(x)
        return s, self.pool(s)


class Up(nn.Module):
    """Nearest-neighbour upsampling x 2, concatenation with the skip (skip first), two Conv blocks."""
    def __init__(self, cin, cout, activation, batch_norm):
        super().__init__()
        self.convs = nn.Sequential(Conv(2 * cin, cout, activation, batch_norm), Conv(cout, cout, activation, batch_norm))

    def forward(self, x, skip):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.convs(torch.cat([skip, x], 1))


class UNet(nn.Module):
    """1 x 1 stem conv -> len(features) - 1 down / up levels with skip connections -> 1 x 1 output conv."""
    def __init__(self, cin, features, cout, activation, batch_norm=True):
        super().__init__()
        f = list(features)
        self.stem = Conv(cin, f[0], activation, batch_norm, kernel_size=1)
        self.down = nn.ModuleList([Down(f[i], f[i + 1], activation, batch_norm) for i in range(len(f) - 1)])
        self.up = nn.ModuleList([Up(f[-i - 1], f[-i - 2], activation, batch_norm) for i in range(len(f) - 1)])
        self.out = nn.Conv2d(f[0], cout, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for d in self.down:
            s, x = d(x)
            skips.append(s)
        for i, u in enumerate(self.up):
            x = u(x, skips[-i - 1])
        return self.out(x)


class SMLMNet(nn.Module):
    """The plain-SMLM detection network (module docstring).

    window: frames per input window (odd; the output belongs to the centre frame). z: 3D (x, y, z) or 2D (x, y).
    cov: None (independent Gaussians per parameter) or 'full_xyz' (a full 3 x 3 covariance of x, y, z; needs z).
    Output scaling: xyz_scale (max offsets: px, px, um), max_bg (photoelectrons / px), max_photons (photoelectrons);
    input_scale / input_offset: the input normalisation (frames - input_offset) x input_scale.
    Widths: frame_features (frame U-Net), window_features (window U-Net), window_out (its output channels),
    head_features (channels per output in the head). batch_norm: BatchNorm2d in the U-Nets."""
    def __init__(self, window=3, z=True, cov=None, max_photons=20000., max_bg=100., xyz_scale=(1., 1., 1.),
                 input_scale=0.01, input_offset=3., frame_features=(48, 96), window_features=(48, 96, 192),
                 window_out=256, head_features=32, batch_norm=True):
        super().__init__()
        if window < 1 or window % 2 == 0:
            raise ValueError(f'window {window}: odd number of frames (the output is the centre frame)')
        if cov not in (None, 'full_xyz'):
            raise ValueError(f"cov {cov!r}: None or 'full_xyz'")
        xyz_scale = [float(v) for v in xyz_scale][:3 if z else 2]
        if len(xyz_scale) < (3 if z else 2) or not all(np.isfinite(s) and s > 0 for s in xyz_scale):
            raise ValueError('xyz_scale: one positive scale per spatial axis')
        if cov and not z:
            raise ValueError("cov='full_xyz' needs z=True")
        self.window, self.z, self.cov = int(window), bool(z), cov
        self.max_photons, self.max_bg, self.xyz_scale = float(max_photons), float(max_bg), xyz_scale
        self.input_scale, self.input_offset = float(input_scale), float(input_offset)
        self.frame_features, self.window_features = list(frame_features), list(window_features)
        self.window_out, self.head_features, self.batch_norm = int(window_out), int(head_features), bool(batch_norm)
        self.params = PARAMS_3D if z else PARAMS_2D
        n = self.n_params = len(self.params)
        scale = [*xyz_scale, self.max_bg, self.max_photons]
        self.register_buffer('output_scale', torch.tensor([1., *scale, *scale], dtype=torch.float32), persistent=False)
        self._tanh = list(range(1, n + 1))
        self._sigmoid = [0, *range(n + 1, 2 * n + 1)]
        self._sigma = list(range(n + 1, 2 * n + 1))
        act = nn.ELU()
        self.frame_unet = UNet(1, self.frame_features, self.frame_features[0], act, batch_norm)
        self.window_unet = UNet(self.frame_features[0] * self.window, self.window_features, self.window_out, act,
                                batch_norm)
        c = 1 + 2 * n
        self.head = nn.Sequential(Conv(self.window_out, head_features * c, nn.ELU(), batch_norm=False, kernel_size=3),
                                  nn.Conv2d(head_features * c, c, kernel_size=1, groups=c))
        self.apply(self._init)
        if cov:
            self.corr_head = nn.Conv2d(self.window_out, 3, 1)
            nn.init.zeros_(self.corr_head.weight)
            nn.init.zeros_(self.corr_head.bias)

    @staticmethod
    def _init(m):
        if type(m) is nn.Conv2d:                     # Kaiming normal, zero bias
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ options
    def options(self):
        """The constructor arguments (model.json 'architecture' and 'scaling' blocks merged)."""
        return dict(window=self.window, z=self.z, cov=self.cov, max_photons=self.max_photons, max_bg=self.max_bg,
                    xyz_scale=list(self.xyz_scale), input_scale=self.input_scale, input_offset=self.input_offset,
                    frame_features=self.frame_features, window_features=self.window_features,
                    window_out=self.window_out, head_features=self.head_features, batch_norm=self.batch_norm)

    @property
    def channels(self):
        return 1 + 2 * self.n_params + (3 if self.cov else 0)

    @property
    def size_multiple(self):
        """Frame height and width must be multiples of this."""
        return 2 ** max(len(self.window_features) - 1, len(self.frame_features) - 1)

    # ------------------------------------------------------------------ forward
    def raw(self, frames):
        """frames [B, window, H, W] (photoelectrons) -> raw outputs [B, C, H, W] (before output scaling)."""
        b, t, h, w = frames.shape
        if t != self.window:
            raise ValueError(f'{t} frames per window, the model has {self.window}')
        x = ((frames.float() - self.input_offset) * self.input_scale).reshape(b * t, 1, h, w)
        x = self.frame_unet(x).reshape(b, -1, h, w)
        x = self.window_unet(x)
        out = self.head(x)
        if self.cov:
            out = torch.cat((out, self.corr_head(x)), 1)
        return out

    def scale(self, raw):
        """raw [B, C, H, W] -> scaled outputs (module docstring) in ROI-local coordinates (x, y offsets of each pixel)."""
        if self.cov:
            rho = 0.95 * raw[:, -3:].float().tanh()
            x = raw[:, :-3].clone()
        else:
            x = raw.clone()
        x[:, self._tanh] = x[:, self._tanh].tanh()
        x[:, self._sigmoid] = x[:, self._sigmoid].sigmoid()
        x[:, self._sigma] = torch.clip(x[:, self._sigma], min=0) + 1e-3
        x = x * self.output_scale[None, :, None, None]
        return torch.cat((x, rho), 1) if self.cov else x

    @staticmethod
    def add_pixel_coordinates(out):
        """Image coordinates: x (channel 1) += column index, y (channel 2) += row index."""
        _, _, h, w = out.shape
        out = out.clone()
        out[:, 2] += torch.arange(h, dtype=torch.float32, device=out.device)[None, :, None]
        out[:, 1] += torch.arange(w, dtype=torch.float32, device=out.device)[None, None, :]
        return out

    def forward(self, frames):
        """frames [B, window, H, W] photoelectrons -> output [B, C, H, W] (module docstring), image coordinates."""
        return self.add_pixel_coordinates(self.scale(self.raw(frames)))

    def loss(self, count_loss_weight=0.01):
        """The training loss of this model (torch_psf.nn.loss.GMMLoss)."""
        from torch_psf.nn.loss import GMMLoss
        return GMMLoss(self.n_params, cov=self.cov, count_loss_weight=count_loss_weight)


def xyz_cholesky(sigma, rho):
    """Cholesky factor L [..., 3, 3] of the x, y, z covariance from the marginal SDs sigma [..., 3] and the partial
    correlations rho [..., 3] = (rho_xy, rho_xz, rho_yz|x): L = diag(sigma) chol(correlation matrix)."""
    a, b, c = rho.unbind(-1)
    u = (1 - a.square()).sqrt()
    v = (1 - b.square()).sqrt()
    zero = torch.zeros_like(a)
    rows = (torch.stack((torch.ones_like(a), zero, zero), -1),
            torch.stack((a, u, zero), -1),
            torch.stack((b, v * c, v * (1 - c.square()).sqrt()), -1))
    return sigma.unsqueeze(-1) * torch.stack(rows, -2)


def correlations(cov):
    """Correlations (xy, xz, yz) [..., 3] of covariance matrices [..., 3, 3]."""
    sd = cov.diagonal(dim1=-2, dim2=-1).sqrt()
    return cov[..., [1, 2, 2], [0, 0, 1]] / (sd[..., [1, 2, 2]] * sd[..., [0, 0, 1]])

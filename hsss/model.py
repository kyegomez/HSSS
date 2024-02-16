import math
from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from zeta.nn import FeedForward


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )

        return output


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(
                Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1])
            )
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


@dataclass
class MambaConfig:
    dim: int  # D
    depth: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = (
            self.expand_factor * self.dim
        )  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.dim / 16)


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.dim)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(
            config.dim, 2 * config.d_inner, bias=config.bias
        )

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(
            config.d_inner,
            config.dt_rank + 2 * config.d_state,
            bias=False,
        )

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(
            config.dt_rank, config.d_inner, bias=True
        )

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(
                self.dt_proj.weight, -dt_init_std, dt_init_std
            )
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(
            1, config.d_state + 1, dtype=torch.float32
        ).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(
            config.d_inner, config.dim, bias=config.bias
        )

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[
            :, :, :L
        ]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC,
            [
                self.config.dt_rank,
                self.config.d_state,
                self.config.d_state,
            ],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0),
            self.config.d_inner,
            self.config.d_state,
            device=deltaA.device,
        )  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # y : (B, D)
        # cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[
            :, :, self.config.d_conv - 1
        ]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # prepare cache for next call
        inputs = torch.cat(
            [inputs[:, :, 1:], x_cache], dim=2
        )  # (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float()
        )  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC,
            [
                self.config.dt_rank,
                self.config.d_state,
                self.config.d_state,
            ],
            dim=-1,
        )  # (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = torch.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
            )  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(
            2
        )  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        # todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.depth)]
        )
        # self.norm_f = RMSNorm(config.dim)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm_f(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class LowLevelMamba(nn.Module):
    """
    LowLevelMamba is a PyTorch module that implements a low-level Mamba model.

    Args:
        dim (int): Dimension of the Mamba model. Default is 4.
        depth (int): Depth of the Mamba model. Default is 3.
        dt_rank (int): Rank of the time step tensor in the Mamba model. Default is 2.
        d_state (int): Dimension of the state tensor in the Mamba model. Default is 2.
        expand_factor (int): Expansion factor of the Mamba model. Default is 2.
        d_conv (int): Dimension of the convolutional kernel in the Mamba model. Default is 3.
        dt_min (float): Minimum value of the time step in the Mamba model. Default is 0.001.
        dt_max (float): Maximum value of the time step in the Mamba model. Default is 0.1.
        dt_init (str): Initialization method for the time step tensor in the Mamba model. Default is "random".
        dt_scale (float): Scaling factor for the time step tensor in the Mamba model. Default is 1.0.
        bias (bool): Whether to include bias terms in the Mamba model. Default is False.
        conv_bias (bool): Whether to include bias terms in the convolutional layers of the Mamba model. Default is True.
        pscan (bool): Whether to use parallel scan operation in the Mamba model. Default is True.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        dim: int = 4,
        depth: int = 3,
        dt_rank: int = 2,
        d_state: int = 2,
        expand_factor: int = 2,
        d_conv: int = 3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan

        config = MambaConfig(
            dim=dim,
            depth=depth,
            dt_rank=dt_rank,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            bias=bias,
            conv_bias=conv_bias,
            pscan=pscan,
            *args,
            **kwargs,
        )

        self.model = Mamba(config)

    def forward(self, x: Tensor):
        """
        Forward pass of the LowLevelMamba model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)


class HighLevelMamba(nn.Module):
    """
    LowLevelMamba is a PyTorch module that implements a low-level Mamba model.

    Args:
        dim (int): Dimension of the Mamba model. Default is 4.
        depth (int): Depth of the Mamba model. Default is 3.
        dt_rank (int): Rank of the time step tensor in the Mamba model. Default is 2.
        d_state (int): Dimension of the state tensor in the Mamba model. Default is 2.
        expand_factor (int): Expansion factor of the Mamba model. Default is 2.
        d_conv (int): Dimension of the convolutional kernel in the Mamba model. Default is 3.
        dt_min (float): Minimum value of the time step in the Mamba model. Default is 0.001.
        dt_max (float): Maximum value of the time step in the Mamba model. Default is 0.1.
        dt_init (str): Initialization method for the time step tensor in the Mamba model. Default is "random".
        dt_scale (float): Scaling factor for the time step tensor in the Mamba model. Default is 1.0.
        bias (bool): Whether to include bias terms in the Mamba model. Default is False.
        conv_bias (bool): Whether to include bias terms in the convolutional layers of the Mamba model. Default is True.
        pscan (bool): Whether to use parallel scan operation in the Mamba model. Default is True.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        dim: int = 4,
        depth: int = 3,
        dt_rank: int = 2,
        d_state: int = 2,
        expand_factor: int = 2,
        d_conv: int = 3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan

        config = MambaConfig(
            dim=dim,
            depth=depth,
            dt_rank=dt_rank,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            bias=bias,
            conv_bias=conv_bias,
            pscan=pscan,
            *args,
            **kwargs,
        )

        self.model = Mamba(config)

    def forward(self, x: Tensor):
        """
        Forward pass of the LowLevelMamba model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)


class SimpleHSSSMamba(nn.Module):
    """
    SimpleHSSSMamba is a PyTorch module that represents the HSSS Mamba model.

    Args:
        dim_in (int): Input dimension for the low level Mamba module.
        depth_in (int): Depth of the low level Mamba module.
        dt_rank_in (int): Rank of the time step tensor for the low level Mamba module.
        d_state_in (int): Dimension of the state tensor for the low level Mamba module.
        expand_factor_in (int): Expansion factor for the low level Mamba module.
        d_conv_in (int): Dimension of the convolutional filters for the low level Mamba module.
        dt_min_in (float): Minimum value for the time step tensor for the low level Mamba module.
        dt_max_in (float): Maximum value for the time step tensor for the low level Mamba module.
        dt_init_in (str): Initialization method for the time step tensor for the low level Mamba module.
        dt_scale_in (float): Scaling factor for the time step tensor for the low level Mamba module.
        bias_in (bool): Whether to include bias in the low level Mamba module.
        conv_bias_in (bool): Whether to include bias in the convolutional filters for the low level Mamba module.
        pscan_in (bool): Whether to use parallel scan in the low level Mamba module.
        dim (int): Input dimension for the high level Mamba module.
        depth (int): Depth of the high level Mamba module.
        dt_rank (int): Rank of the time step tensor for the high level Mamba module.
        d_state (int): Dimension of the state tensor for the high level Mamba module.
        expand_factor (int): Expansion factor for the high level Mamba module.
        d_conv (int): Dimension of the convolutional filters for the high level Mamba module.
        dt_min (float): Minimum value for the time step tensor for the high level Mamba module.
        dt_max (float): Maximum value for the time step tensor for the high level Mamba module.
        dt_init (str): Initialization method for the time step tensor for the high level Mamba module.
        dt_scale (float): Scaling factor for the time step tensor for the high level Mamba module.
        bias (bool): Whether to include bias in the high level Mamba module.
        conv_bias (bool): Whether to include bias in the convolutional filters for the high level Mamba module.
        pscan (bool): Whether to use parallel scan in the high level Mamba module.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim_in: int = 8,
        depth_in: int = 6,
        dt_rank_in: int = 4,
        d_state_in: int = 4,
        expand_factor_in: int = 4,
        d_conv_in: int = 6,
        dt_min_in: float = 0.001,
        dt_max_in: float = 0.1,
        dt_init_in: str = "random",
        dt_scale_in: float = 1.0,
        bias_in: bool = False,
        conv_bias_in: bool = True,
        pscan_in: bool = True,
        dim: int = 4,
        depth: int = 3,
        dt_rank: int = 2,
        d_state: int = 2,
        expand_factor: int = 2,
        d_conv: int = 3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True,
        proj_layer: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.depth_in = depth_in
        self.dt_rank_in = dt_rank_in
        self.d_state_in = d_state_in
        self.expand_factor_in = expand_factor_in
        self.d_conv_in = d_conv_in
        self.dt_min_in = dt_min_in
        self.dt_max_in = dt_max_in
        self.dt_init_in = dt_init_in
        self.dt_scale_in = dt_scale_in
        self.bias_in = bias_in
        self.conv_bias_in = conv_bias_in
        self.pscan_in = pscan_in
        self.dim = dim
        self.depth = depth
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan

        self.low_level_mamba = LowLevelMamba(
            dim=dim_in,
            depth=depth_in,
            dt_rank=dt_rank_in,
            d_state=d_state_in,
            expand_factor=expand_factor_in,
            d_conv=d_conv_in,
            dt_min=dt_min_in,
            dt_max=dt_max_in,
            dt_init=dt_init_in,
            dt_scale=dt_scale_in,
            bias=bias_in,
            conv_bias=conv_bias_in,
            pscan=pscan_in,
        )

        if proj_layer:
            # Linear projection
            self.ffn = FeedForward(
                dim_in, dim, dim * 4, *args, **kwargs
            )
        else:
            self.ffn = nn.Linear(dim_in, dim)

        # High level Mamba
        self.high_level_mamba = HighLevelMamba(
            dim=dim,
            depth=depth,
            dt_rank=dt_rank,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            bias=bias,
            conv_bias=conv_bias,
            pscan=pscan,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SimpleHSSSMamba model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.low_level_mamba(x)
        x = self.ffn(x)
        x = self.high_level_mamba(x)
        return x


class HSSS(nn.Module):
    """
    High-level Sparse State Space (HSSS) model.

    Args:
        layers (List[nn.Module]): List of layers to be applied to the input.
        dim (int): Dimensionality of the output tensor (default: 4).
        depth (int): Number of layers in the HSSS model (default: 3).
        dt_rank (int): Rank of the time step tensor (default: 2).
        d_state (int): Dimensionality of the state tensor (default: 2).
        expand_factor (int): Expansion factor for the state tensor (default: 2).
        d_conv (int): Dimensionality of the convolutional kernel (default: 3).
        dt_min (float): Minimum value for the time step tensor (default: 0.001).
        dt_max (float): Maximum value for the time step tensor (default: 0.1).
        dt_init (str): Initialization method for the time step tensor (default: "random").
        dt_scale (float): Scaling factor for the time step tensor (default: 1.0).
        bias (bool): Whether to include bias terms in the model (default: False).
        conv_bias (bool): Whether to include bias terms in the convolutional layers (default: True).
        pscan (bool): Whether to use parallel scan for the convolutional layers (default: True).
        proj_layer (bool): Whether to include a projection layer in the model (default: True).
        ffn (bool): Whether to use a feed-forward network for the final layer (default: True).
        dropout (float): Dropout probability for the feed-forward network (default: 0.1).
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        layers: List[nn.Module],
        dim: int = 4,
        depth: int = 3,
        dt_rank: int = 2,
        d_state: int = 2,
        expand_factor: int = 2,
        d_conv: int = 3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True,
        proj_layer: bool = True,
        ffn: bool = True,
        dropout: float = 0.1,
        dim_range: int = -2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.layers = layers
        self.dim = dim
        self.depth = depth
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.bias = bias
        self.conv_bias = conv_bias
        self.pscan = pscan
        self.proj_layer = proj_layer
        self.ffn = ffn
        self.ffn_mult = dim * 4
        self.dropout = dropout
        self.dim_range = dim_range

        # low level mambas
        self.high_level_mamba = HighLevelMamba(
            dim=dim,
            depth=depth,
            dt_rank=dt_rank,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            bias=bias,
            conv_bias=conv_bias,
            pscan=pscan,
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor, *args, **kwargs):
        """
        Forward pass of the HSSS model.

        Args:
            x (Tensor): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Output tensor of the HSSS model.
        """
        print(x.shape)
        b, s, d = x.shape

        layer_outputs = []
        for layer in self.layers:
            layer_output = layer(x)
            print(x.shape)
            layer_outputs.append(layer_output)

        x = torch.cat(layer_outputs, dim=self.dim_range)

        if self.ffn:
            x = FeedForward(
                x.size(-1),
                self.dim,
                self.ffn_mult,
            )(x)
            print(x.shape)
        else:
            x = nn.Linear(x.size(-1), self.dim)(x)

        x =  self.high_level_mamba(x)
        
        x = torch.split(
            x,
            len(self.layers),
            dim=self.dim_range,
        )
        
        return x
[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# HSSS
Implementation of a Hierarchical Mamba as described in the paper: "Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling".


## install
`pip install hsss`

##  usage
```python
import torch
from hsss.model import HSSSMamba

x = torch.randn(1, 10, 8)

model = HSSSMamba(
    dim_in=8,  # dimension of input
    depth_in=6,  # depth of input
    dt_rank_in=4,  # rank of input
    d_state_in=4,  # state of input
    expand_factor_in=4,  # expansion factor of input
    d_conv_in=6,  # convolution dimension of input
    dt_min_in=0.001,  # minimum time step of input
    dt_max_in=0.1,  # maximum time step of input
    dt_init_in="random",  # initialization method of input
    dt_scale_in=1.0,  # scaling factor of input
    bias_in=False,  # whether to use bias in input
    conv_bias_in=True,  # whether to use bias in convolution of input
    pscan_in=True,  # whether to use parallel scan in input
    dim=4,  # dimension of model
    depth=3,  # depth of model
    dt_rank=2,  # rank of model
    d_state=2,  # state of model
    expand_factor=2,  # expansion factor of model
    d_conv=3,  # convolution dimension of model
    dt_min=0.001,  # minimum time step of model
    dt_max=0.1,  # maximum time step of model
    dt_init="random",  # initialization method of model
    dt_scale=1.0,  # scaling factor of model
    bias=False,  # whether to use bias in model
    conv_bias=True,  # whether to use bias in convolution of model
    pscan=True,  # whether to use parallel scan in model
)

out = model(x)
print(out)
```

# License
MIT

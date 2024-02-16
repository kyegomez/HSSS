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
    dim_in = 8,
    depth_in = 6,
    dt_rank_in = 4,
    d_state_in = 4,
    expand_factor_in = 4,
    d_conv_in = 6,
    dt_min_in = 0.001,
    dt_max_in = 0.1,
    dt_init_in = "random",
    dt_scale_in = 1.0,
    bias_in = False,
    conv_bias_in = True,
    pscan_in = True,
    dim = 4,
    depth = 3,
    dt_rank = 2,
    d_state = 2,
    expand_factor = 2,
    d_conv = 3,
    dt_min = 0.001,
    dt_max = 0.1,
    dt_init = "random",
    dt_scale = 1.0,
    bias = False,
    conv_bias = True,
    pscan = True,
)


out = model(x)
print(out)
```

# License
MIT

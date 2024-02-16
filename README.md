[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# HSSS
Implementation of a Hierarchical Mamba as described in the paper: "Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling" but instead of using traditional SSMs were using Mambas. Basically the flow is single input -> low level mambas -> concat -> high level ssm -> multiple outputs. 


[Paper link](https://arxiv.org/pdf/2402.10211.pdf)


I believe in this architecture alot as it segments local and global learning. 


## install
`pip install hsss`

##  usage
```python
import torch
from hsss import LowLevelMamba, HSSS


# Reandom tensor
x = torch.randn(1, 10, 8)

# Low level model
mamba = LowLevelMamba(
    dim=8,  # dimension of input
    depth=6,  # depth of input
    dt_rank=4,  # rank of input
    d_state=4,  # state of input
    expand_factor=4,  # expansion factor of input
    d_conv=6,  # convolution dimension of input
    dt_min=0.001,  # minimum time step of input
    dt_max=0.1,  # maximum time step of input
    dt_init="random",  # initialization method of input
    dt_scale=1.0,  # scaling factor of input
    bias=False,  # whether to use bias in input
    conv_bias=True,  # whether to use bias in convolution of input
    pscan=True,  # whether to use parallel scan in input
)


# Low level model 2
mamba2 = LowLevelMamba(
    dim=8,  # dimension of input
    depth=6,  # depth of input
    dt_rank=4,  # rank of input
    d_state=4,  # state of input
    expand_factor=4,  # expansion factor of input
    d_conv=6,  # convolution dimension of input
    dt_min=0.001,  # minimum time step of input
    dt_max=0.1,  # maximum time step of input
    dt_init="random",  # initialization method of input
    dt_scale=1.0,  # scaling factor of input
    bias=False,  # whether to use bias in input
    conv_bias=True,  # whether to use bias in convolution of input
    pscan=True,  # whether to use parallel scan in input
)


# Low level mamba 3
mamba3 = LowLevelMamba(
    dim=8,  # dimension of input
    depth=6,  # depth of input
    dt_rank=4,  # rank of input
    d_state=4,  # state of input
    expand_factor=4,  # expansion factor of input
    d_conv=6,  # convolution dimension of input
    dt_min=0.001,  # minimum time step of input
    dt_max=0.1,  # maximum time step of input
    dt_init="random",  # initialization method of input
    dt_scale=1.0,  # scaling factor of input
    bias=False,  # whether to use bias in input
    conv_bias=True,  # whether to use bias in convolution of input
    pscan=True,  # whether to use parallel scan in input
)


# HSSS
hsss = HSSS(
    layers=[mamba, mamba2, mamba3],
    dim=12,  # dimension of model
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
    proj_layer=True,
)


# Forward pass
out = hsss(x)
print(out)

```
## Citation
```bibtex
@misc{bhirangi2024hierarchical,
      title={Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling}, 
      author={Raunaq Bhirangi and Chenyu Wang and Venkatesh Pattabiraman and Carmel Majidi and Abhinav Gupta and Tess Hellebrekers and Lerrel Pinto},
      year={2024},
      eprint={2402.10211},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


# License
MIT


## Todo 

- [ ] Implement the chunking of the tokens by spliting it up the sequence dimension

- [ ] Make the fusion projection layer dynamic and not use just a linear, ffn, or cross attention or even an output head.

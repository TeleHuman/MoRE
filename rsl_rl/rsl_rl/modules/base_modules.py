import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import init

class MLPBase(nn.Module):
  def __init__(
      self,
      input_shape,
      hidden_shapes,
      activation_func=nn.ReLU,
      init_func=init.basic_init,
      add_ln=False,
      last_activation_func=None):
    super().__init__()

    self.activation_func = activation_func
    self.fcs = []
    self.add_ln = add_ln
    if last_activation_func is not None:
      self.last_activation_func = last_activation_func
    else:
      self.last_activation_func = activation_func
    input_shape = np.prod(input_shape)

    self.output_shape = input_shape
    for next_shape in hidden_shapes:
      fc = nn.Linear(input_shape, next_shape)
      init_func(fc)
      self.fcs.append(fc)
      self.fcs.append(activation_func())
      if self.add_ln:
        self.fcs.append(nn.LayerNorm(next_shape))
      input_shape = next_shape
      self.output_shape = next_shape

    self.fcs.pop(-1) # Remove last activation function
    self.fcs.append(self.last_activation_func())
    self.seq_fcs = nn.Sequential(*self.fcs)

  def forward(self, x):
    return self.seq_fcs(x)

# From https://github.com/joonleesky/train-procgen-pytorch/blob/1678e4a9e2cb8ffc3772ecb3b589a3e0e06a2281/common/model.py#L94

def xavier_uniform_init(module, gain=1.0):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.xavier_uniform_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module

class Flatten(nn.Module):
  def forward(self, x):
    # base_shape = v.shape[-2]
    return x.view(x.size(0), -1)

def weight_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
  elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class RLProjection(nn.Module):
  def __init__(self, in_dim, out_dim, proj=True):
    super().__init__()
    self.out_dim = out_dim
    module_list = [
      nn.Linear(in_dim, out_dim)
    ]
    if proj:
      module_list += [
        # nn.LayerNorm(out_dim),
        # nn.Tanh()
        nn.ReLU()
      ]

    self.projection = nn.Sequential(
      *module_list
    )
    self.output_dim = out_dim
    self.apply(weight_init)

  def forward(self, x):
    return self.projection(x)

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module

class NatureEncoder(nn.Module):
  def __init__(self,
               in_channels,
               groups=1,
               flatten=True,
               **kwargs):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super(NatureEncoder, self).__init__()
    self.groups = groups
    layer_list = [
      nn.Conv2d(in_channels=in_channels, out_channels=32 * self.groups,
                kernel_size=8, stride=4), nn.ReLU(),
      nn.Conv2d(in_channels=32 * self.groups, out_channels=64 * self.groups,
                kernel_size=4, stride=2), nn.ReLU(),
      nn.Conv2d(in_channels=64 * self.groups, out_channels=64 * self.groups,
                kernel_size=3, stride=1), nn.ReLU(),
    ]
    if flatten:
      layer_list.append(
        Flatten()
      )
    self.layers = nn.Sequential(*layer_list)

    self.output_dim = 1024 * self.groups
    self.apply(orthogonal_init)

  def forward(self, x, detach=False):
    # view_shape = x.size()[:-3] + torch.Size([-1])
    x = x.view(torch.Size(
      [np.prod(x.size()[:-3])]) + x.size()[-3:])
    x = self.layers(x)
    # x = x.view(view_shape)
    if detach:
      x = x.detach()
    return x

class StateEncoder(nn.Module):
  def __init__(
      self,
      state_input_dim,
      hidden_shapes,
      token_dim=64,
      **kwargs
  ):
    super(StateEncoder, self).__init__()
    # assert in_channels == 16
    self.token_dim = token_dim
    # self.depth_visual_base = NatureEncoder(2, flatten=False)

    # self.depth_up_conv = nn.Conv2d(64, token_dim, 1)

    self.base = MLPBase(input_shape=state_input_dim, hidden_shapes=hidden_shapes,**kwargs)
    
    self.state_projector = RLProjection(in_dim=self.base.output_shape, out_dim=token_dim)
    
    # self.per_modal_tokens = 16

  def forward(self, state_x, detach=False):
    state_out = self.base(state_x)

    state_out_proj = self.state_projector(state_out)
    state_out_proj = state_out_proj.unsqueeze(0)

    out_list = [state_out_proj]
    output = torch.cat(out_list, dim=0)

    return output

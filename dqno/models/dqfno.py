from typing import Tuple, List, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbedding2D
from ..layers.channel_mlp import ChannelMLP, LinearChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_block import FNOBlocks
from ..layers.derived import DerivedMLP
from .base_model import BaseModel

class DQFNO(BaseModel, name='DQFNO'):
    # TODO - Implement more doc strings
    """3-Dimensional Derived Quanitity Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions, additionally 
    it learns the mappings of these functions to their physically based derived quanities.
    
    The key component of an DQFNO is its SpectralConv layer (see 
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN 
    conv layer but operates in the frequency domain.

    Parameters
    ----------
    n_modes : List[Tuple]
        number of modes to keep in Fourier Layer, along each variable and 
        dimension. The dimensionality of the FNO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the DQFNO (i.e. number of channels), by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4"""
    def __init__(
        self,
        modes: List[Tuple[int,int]],
        in_channels: int,
        out_channels: int,
        dx: float,
        hidden_channels: int = 256,
        n_layers: int = 4,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        conv_module: nn.Module = SpectralConv,
        derived_module: nn.Module = DerivedMLP,
        derived_type: str = "none",
        debug: bool = False,
        **kwargs
    ) -> None:

        super().__init__()
        self.debug = debug
        self.n_dim = len(modes)
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dx = dx
        self.n_layers = n_layers
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * hidden_channels
        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * hidden_channels
        self.non_linearity = non_linearity
        self._modes = modes
        
        if positional_embedding == "grid":
            self.positional_embedding = GridEmbedding2D()
        else:
            raise ValueError(f"Invalid positional embedding: {positional_embedding}. Expected 'grid' or an nn.Module.")

        lifting_in_channels = self.in_channels + 2 if self.positional_embedding else self.in_channels
        self.lifting = ChannelMLP(lifting_in_channels, self.hidden_channels) 
        
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            modes=self.modes,
            non_linearity=non_linearity,
            conv_module=conv_module,
            n_layers=n_layers,
            debug = self.debug,
            **kwargs
        )
        
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

        self.derived_type = derived_type
        self.derived_module = DerivedMLP(
            _type=self.derived_type,
            dx = dx,
        ) if derived_module != None else None
        
        if self.debug:
            self._debug_print_initialization()

    def forward(self, x: torch.Tensor, output_shape: Union[Tuple[int, ...], List[Tuple[int, ...]], None] = None, **kwargs) -> torch.Tensor:
        x, derived = x # derived shape (B, D)
        b, c, t, v, h, w = x.shape
        
        if self.derived_module != None:
            self.derived_module.store(derived)

        if self.debug:
            print(f"Input shape: (batch={b}, channels={c}, timesteps={t}, vars={v}, height={h}, width={w})")

        output_shape = [None] * self.n_layers if output_shape is None else output_shape if isinstance(output_shape, list) else [None] * (self.n_layers - 1) + [output_shape]
        
        if self.positional_embedding:
            x = self.positional_embedding(x)
        
        x = self.lifting(x)
        
        if self.debug:
            print(f"Shape after lifting: {x.shape}")
        
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
            if self.debug:
                print(f"Shape after FNO layer {layer_idx + 1}: {x.shape}")
        
        x = self.projection(x)

        if self.debug:
            self._debug_print_model_parameters()
        
        if self.derived_module != None:
            derived_x = self.derived_module(x)
            return (x, derived_x)
        
        if self.derived_module == None:
            return x

    
    @property
    def modes(self) -> Tuple[int, ...]:
        return self._modes

    @modes.setter
    def modes(self, modes: Tuple[int, ...]) -> None:
        self.fno_blocks.modes = modes
        self._modes = modes
    
    def _debug_print_initialization(self) -> None:
        print("[DEBUG] DQFNO Model Initialization")
        print(f"Modes: {self.modes}")
        print(f"In Channels: {self.in_channels}, Out Channels: {self.out_channels}")
        print(f"Hidden Channels: {self.hidden_channels}, Number of Layers: {self.n_layers}")
        print(f"Lifting Channels: {self.lifting_channels}, Projection Channels: {self.projection_channels}")
        print(f"Using Non-linearity: {self.non_linearity}")
        print(f"Positional Embedding: {self.positional_embedding}")
    
    def _debug_print_model_parameters(self) -> None:
        print("[DEBUG] Model Parameters:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    def save(self, directory: str, filename: str = "model.pth") -> None:
        """
        Saves the model's state dictionary and hyperparameters to the specified directory.

        Args:
            directory (str): Path to the directory where the model will be saved.
            filename (str, optional): Name of the file to save the model as. Defaults to "model.pth".
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        checkpoint = {
            "state_dict": self.state_dict(),
            "hyperparameters": {
                "modes": self._modes,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "dx": self.dx,
                "hidden_channels": self.hidden_channels,
                "n_layers": self.n_layers,
                "lifting_channel_ratio": self.lifting_channel_ratio,
                "projection_channel_ratio": self.projection_channel_ratio,
                "positional_embedding": "grid" if isinstance(self.positional_embedding, GridEmbedding2D) else None,
                "derived_type": self.derived_type, 
                "non_linearity": self.non_linearity,
                "debug": self.debug,
            }
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, directory: str, filename: str = "model.pth", device: torch.device = torch.device("cpu")) -> "DQFNO":
        """
        Loads the model from a checkpoint, automatically restoring hyperparameters.

        Args:
            directory (str): Path to the directory containing the model checkpoint.
            filename (str, optional): Name of the file to load. Defaults to "model.pth".
            device (torch.device, optional): Device to load the model onto. Defaults to CPU.

        Returns:
            DQFNO: The loaded model instance.
        """
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        model = cls(**checkpoint["hyperparameters"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        print(f"Model loaded from {filepath} with restored hyperparameters.")
        return model
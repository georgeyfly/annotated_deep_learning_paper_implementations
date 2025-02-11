import torch
from torch import nn

from labml_helpers.module import Module

class InstanceNorm(Module):
    """
    Instance Normalization Layer Implementation
        
    Instance normalization performs feature normalization for each channel of each instance
    in the batch independently, making it particularly useful for style transfer and 
    image generation tasks. Unlike BatchNorm which normalizes across the batch dimension,
    and LayerNorm which normalizes across all features, InstanceNorm normalizes across
    spatial dimensions (H,W) for each channel independently.
        
    For an input tensor X ∈ ℝ^(B×C×H×W):
        B: batch size (each instance normalized independently)
        C: number of channels (each channel normalized independently)
        H: height (spatial dimension to normalize over)
        W: width (spatial dimension to normalize over)
        
    Mathematical formulation:
    IN(x) = γ * ((x - E[x]) / sqrt(Var[x] + ε)) + β
        
    where for each channel c and instance b:
        E[x] is computed across H,W dimensions only
        Var[x] is computed across H,W dimensions only
        γ (scale) and β (shift) are learnable parameters per channel
        ε is a small constant for numerical stability
        
    Key differences from other normalizations:
    - BatchNorm: normalizes across (B,H,W), tracking running statistics
    - LayerNorm: normalizes across (C,H,W)
    - InstanceNorm: normalizes across (H,W) only, independently for each (B,C)
    """

    def __init__(self, channels: int, *,
                eps: float = 1e-5, 
                affine: bool = True):
        """
        Initialize the InstanceNorm layer.
            
        Args:
            channels (int): Number of input channels (C)
            eps (float): Small constant for numerical stability in division
            affine (bool): If True, apply learnable affine transformation
            
        Returns:
            None
                
        TODO:
        1. Initialize parent class using super()
            
        2. Store instance parameters:
        - channels: number of input channels
        - eps: numerical stability constant
        - affine: whether to use learnable transformation
            
        3. If affine is True:
        - Create learnable parameter 'scale' (γ) of shape [channels]
        - Initialize scale to ones
        - Create learnable parameter 'shift' (β) of shape [channels]
        - Initialize shift to zeros
            
        Note: Unlike BatchNorm, InstanceNorm doesn't track running statistics
        since it normalizes each instance independently at both train and test time.
        """
        # Initialize parent class
        super().__init__()

        # Store parameters
        self.channels = channels
        self.eps = eps
        self.affine = affine

        # TODO: Initialize learnable parameters if affine is True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply instance normalization to the input tensor.
            
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, *]
                where * represents spatial dimensions (e.g., height, width)
                Common shapes:
                - 2D images: [batch_size, channels, height, width]
                - 1D sequences: [batch_size, channels, length]
            
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input
                
        TODO:
        1. Input validation and preparation:
        - Store original input shape
        - Verify channels dimension matches self.channels
        - Reshape input to [batch_size, channels, -1] to handle arbitrary
            spatial dimensions uniformly
            
        2. Statistics computation (for each instance and channel independently):
        - Compute mean across spatial dimensions (last dimension after reshape)
        - Keep dimensions for broadcasting
        - Compute mean of squared values across spatial dimensions
        - Calculate variance using E[x²] - (E[x])²
            
        3. Normalization:
        - Subtract mean from input (broadcasting across spatial dimensions)
        - Divide by sqrt(variance + eps)
            
        4. Affine transformation (if enabled):
        - Reshape scale parameter to [1, channels, 1] for broadcasting
        - Reshape shift parameter to [1, channels, 1] for broadcasting
        - Apply scale and shift to normalized values
            
        5. Reshape and return:
        - Restore original input shape
        - Return normalized tensor
            
        Note: Each instance and channel is normalized independently, making this
        operation suitable for style transfer where we want to normalize content
        statistics while preserving style information across channels.
        """
        # Store original shape and validate input
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]

        # TODO: Implement instance normalization logic

        return None  # Replace with actual normalized tensor


def _test():
    """
    Simple test
    """
    from labml.logger import inspect

    x = torch.zeros([2, 6, 2, 4])
    inspect(x.shape)
    bn = InstanceNorm(6)

    x = bn(x)
    inspect(x.shape)


#
if __name__ == "__main__":
    _test()

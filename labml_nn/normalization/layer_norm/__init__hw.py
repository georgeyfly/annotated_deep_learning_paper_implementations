class LayerNorm(Module):
    """
    Layer Normalization Implementation
        
    This layer normalizes input features across the feature dimensions (not batch dimension)
    to stabilize training. It supports multiple input formats:
        
    1. Linear input: (B, C)
    - B: batch size
    - C: number of features
    - Normalizes across C dimension
        
    2. Sequence input: (L, B, C)
    - L: sequence length
    - B: batch size  
    - C: number of features
    - Normalizes across C dimension
        
    3. Image input: (B, C, H, W) 
    - B: batch size
    - C: channels
    - H: height
    - W: width
    - Normalizes across (C,H,W) dimensions
    - Note: This is not a common use case
        
    Mathematical formulation:
    LN(x) = γ * ((x - E[x]) / sqrt(Var[x] + ε)) + β
        
    where:
        γ (gain) and β (bias) are learnable parameters
        E[x] is the mean computed over the normalized dimensions
        Var[x] is the variance computed over the normalized dimensions
        ε is a small constant for numerical stability
        
    Key difference from BatchNorm:
        - LayerNorm normalizes across feature dimensions
        - BatchNorm normalizes across batch dimension
    """

    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                eps: float = 1e-5,
                elementwise_affine: bool = True):
        """
        Initialize the LayerNorm layer.
            
        Args:
            normalized_shape (Union[int, List[int], Size]): Shape of features to normalize over
                - For linear: [features]
                - For sequences: [features]
                - For images: [channels, height, width]
            eps (float): Small constant for numerical stability
            elementwise_affine (bool): If True, use learnable affine parameters
            
        Returns:
            None
                
        TODO:
        1. Initialize parent class
            
        2. Process normalized_shape:
        - If integer: convert to torch.Size([normalized_shape])
        - If list: convert to torch.Size(normalized_shape)
        - Validate it's a torch.Size object
            
        3. Store instance parameters:
        - normalized_shape
        - eps
        - elementwise_affine
            
        4. If elementwise_affine is True:
        - Create learnable parameter 'gain' (γ) with shape normalized_shape
        - Initialize gain to ones
        - Create learnable parameter 'bias' (β) with shape normalized_shape
        - Initialize bias to zeros
        """
        # Initialize parent class
        super().__init__()
            
        # Store basic parameters
        self.eps = eps
        self.elementwise_affine = elementwise_affine
            
        # TODO: Process normalized_shape and initialize parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.
            
        Args:
            x (torch.Tensor): Input tensor of shape [*, S[0], S[1], ..., S[n]]
                where * represents any number of leading dimensions and
                S[i] are the normalized dimensions
                    
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input
                
        TODO:
        1. Input validation:
        - Verify that the last dimensions match self.normalized_shape
        - Handle arbitrary number of leading dimensions
            
        2. Calculate normalization dimensions:
        - Create list of dimensions to normalize over
        - These should be the last N dimensions where N = len(normalized_shape)
            
        3. Statistics computation:
        - Compute mean across normalization dimensions (keep dims)
        - Compute variance across normalization dimensions:
            * Calculate mean of squared values
            * Subtract squared mean from mean of squares
            * Keep dimensions for broadcasting
            
        4. Normalization:
        - Subtract mean from input
        - Divide by sqrt(variance + eps)
            
        5. Scale and shift (if elementwise_affine):
        - Apply gain parameter (γ)
        - Apply bias parameter (β)
            
        6. Return normalized tensor
        """
        # Validate input shape matches normalized_shape
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
            
        # TODO: Implement layer normalization logic
            
        return None  # Replace with actual normalized tensor
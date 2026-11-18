class MariaConfig:
    vocab_size: int              = 128_256
    hidden_size: int             = 2048
    intermediate_size: int       = 5632
    num_hidden_layers: int       = 18
    num_attention_heads: int     = 16
    num_key_value_heads: int     = 4
    max_position_embeddings: int = 8192
    rope_theta: float            = 500_000.0
    attention_bias: bool         = False
    layer_norm_eps: float        = 1e-5
    dropout: float               = 0.0
    tie_weights: bool            = True
    gradient_checkpointing: bool = True  # ENABLED for 8k long context

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

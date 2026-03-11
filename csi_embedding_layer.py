class CSIEmbeddingLayer(nn.Module):
    """
    replace Embedding layer of DeepSeek precisly
    map CSI data to hiding layer of model
    """

    def __init__(self, config, csi_input_dim: int = 128):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # feature coding of CSI (replaced by CNN/trasformer etc)
        self.csi_encoder = nn.Sequential(
            nn.Linear(csi_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        # position coding(multiplex the position coding of DeepSeek or be customed)
        # learnable position coding
        self.position_embeddings = nn.Embedding(1024, self.hidden_size)  # max length of vector is 1024

        # initialize weight
        self._init_weights()

    def _init_weights(self):
        """weight initialization"""
        for module in self.csi_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, csi_input: torch.Tensor) -> torch.Tensor:
        """
        forward propagation
        Args:
            csi_input: [batch_size, seq_len, csi_input_dim]
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = csi_input.shape

        # 1. featur extractiong by CSI encoder
        csi_features = self.csi_encoder(csi_input)  # [batch, seq_len, hidden_size]

        # 2. add position coding
        position_ids = torch.arange(seq_len, dtype=torch.long, device=csi_input.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len]
        position_embeddings = self.position_embeddings(position_ids)  # [batch, seq_len, hidden_size]

        # 3. combine fearure and position组合特征和位置编码
        hidden_states = csi_features + position_embeddings

        # 4. LayerNorm (可选，如果DeepSeek模型有)
        # hidden_states = self.LayerNorm(hidden_states)

        return hidden_states
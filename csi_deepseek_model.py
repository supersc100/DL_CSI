import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import csi_embedding_layer
class CSIDeepSeekModel(nn.Module):
    """
    CSI prediction model based on DeepSeek
    function:predicate UL CSI by inputing DL CSI
    """

    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
                 csi_input_dim: int = 128,
                 csi_output_dim: int = 128,
                 use_lora: bool = True,
                 lora_r: int = 8):
        super().__init__()

        print(f"Loading Model: {model_name}")

        # 1. load configuration of ds model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # 2. load ds model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # 3. frozen parameters of base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("parameters of base model is frozen")

        # 4. replace Embdding layer
        self.csi_embedding = csi_embedding_layer.CSIEmbeddingLayer(self.config, csi_input_dim)
        self.base_model.model.embed_tokens = self.csi_embedding

        # 5. fine-tuning by LoRA
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,  # LoRA rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"LoRA已启用 (r={lora_r})")

        # 6. replace output layer: CSI regression head
        # origin output dimension -> UL CSI dimension
        hidden_size = self.config.hidden_size
        self.csi_regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, csi_output_dim)
        )

        # 7. print info of model
        self._print_model_info()

    def _print_model_info(self):
        """print info of model paras"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Statics of Model Para:")
        print(f"  Total Para: {total_params:,}")
        print(f"  Trainable Para: {trainable_params:,}")
        print(f"  Trainable Para Rate: {trainable_params / total_params * 100:.2f}%")

    def forward(self, downlink_csi: torch.Tensor) -> torch.Tensor:
        """
        forward propagation：DL CSI -> UL CSI prediction

        Args:
            downlink_csi: DL CSI Data [batch, seq_len, csi_input_dim]

        Returns:
            uplink_pred: UL CSI predicted [batch, seq_len, csi_output_dim]
        """
        # 1. CSI Embedding layer
        inputs_embeds = self.csi_embedding(downlink_csi)

        # 2. DeepSeek Transformer
        transformer_outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,  # provide mask if vector has padding
            output_hidden_states=True
        )

        # 3. get status of the last hidden layer
        last_hidden_state = transformer_outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # 4. CSI regression head
        uplink_pred = self.csi_regression_head(last_hidden_state)  # [batch, seq_len, csi_output_dim]

        return uplink_pred

    def save_model(self, path: str):
        """save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
        print(f"model is save to: {path}")

    def load_model(self, path: str):
        """load model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"model is loaded from {path}")
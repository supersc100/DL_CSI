import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Optional

#check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device using: {device}")

# path for saving
os.makedirs("csi_models", exist_ok=True)
os.makedirs("csi_data", exist_ok=True)
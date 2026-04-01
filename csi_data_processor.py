import torch
import numpy as np
from typing import Tuple

class CSIDataProcessor:
    """
    CSI data processor
    Format of CSI data: [batch, num_antennas, num_subcarriers, 2] (real and imag)
    """

    def __init__(self, seq_len: int = 64, num_antennas: int = 4, num_subcarriers: int = 64):
        self.seq_len = seq_len
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.input_dim = num_antennas * num_subcarriers * 2  # real and imag

    def preprocess_csi(self, csi_complex: np.ndarray) -> torch.Tensor:
        """
        pre-proc CSI complex data
        Args:
            csi_complex: [batch, antennas, subcarriers]
        Returns:
            tensor after proc [batch, seq_len, input_dim]
        """
        batch_size = csi_complex.shape[0]

        # separate real and imag
        csi_real = np.real(csi_complex).reshape(batch_size, -1)
        csi_imag = np.imag(csi_complex).reshape(batch_size, -1)

        # concatenate and normalization
        csi_combined = np.concatenate([csi_real, csi_imag], axis=1)

        # normalize to [-1, 1]
        csi_max = np.max(np.abs(csi_combined), axis=1, keepdims=True) + 1e-8
        csi_normalized = csi_combined / csi_max

        # reshape the format of vector [batch, seq_len, features_per_step]
        features_per_step = self.input_dim // self.seq_len
        csi_reshaped = csi_normalized.reshape(batch_size, self.seq_len, features_per_step)

        return torch.FloatTensor(csi_reshaped)

    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        generate combining CSI data to test
        DL CSI -> UL CSI mapping
        """
        print(f"generate {num_samples} CSI samples...")

        # generate DL CSI (input)
        downlink_csi = []
        # generate UL CSI (target，different with DL CSI but has relation)
        uplink_csi = []

        for i in range(num_samples):
            # generate random channel matrix (Relay Fading)
            H_down = np.random.randn(self.num_antennas, self.num_subcarriers) + \
                     1j * np.random.randn(self.num_antennas, self.num_subcarriers)

            # reciprocity between UL CSI and DL CSI
            H_up = H_down * np.exp(1j * np.random.uniform(-0.3, 0.3, size=H_down.shape)) * \
                   np.random.uniform(0.8, 1.2, size=H_down.shape)

            downlink_csi.append(H_down)
            uplink_csi.append(H_up)

        # transform to tensor
        downlink_tensor = self.preprocess_csi(np.array(downlink_csi))
        uplink_tensor = self.preprocess_csi(np.array(uplink_csi))

        return downlink_tensor, uplink_tensor
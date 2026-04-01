import torch
import torch.nn as nn
import torch.nn.functional as F

class CSILoss(nn.Module):
    """
    definition loss function of CSI prediction
    """

    def __init__(self,
                 mse_weight: float = 1.0,
                 phase_weight: float = 0.5,
                 correlation_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.phase_weight = phase_weight
        self.correlation_weight = correlation_weight

    def complex_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """complex MSE loss"""
        # Assume expanded real and imag as input
        batch_size, seq_len, features = pred.shape
        # reconstruct to complex [batch, seq_len, features//2]
        pred_real = pred[:, :, :features // 2]
        pred_imag = pred[:, :, features // 2:]
        target_real = target[:, :, :features // 2]
        target_imag = target[:, :, features // 2:]

        mse_real = F.mse_loss(pred_real, target_real)
        mse_imag = F.mse_loss(pred_imag, target_imag)

        return (mse_real + mse_imag) / 2

    def phase_cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Phase cosine similarity loss"""
        batch_size, seq_len, features = pred.shape
        pred_real = pred[:, :, :features // 2]
        pred_imag = pred[:, :, features // 2:]
        target_real = target[:, :, :features // 2]
        target_imag = target[:, :, features // 2:]

        # calculate magnitude and phase
        pred_magnitude = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + 1e-8)
        target_magnitude = torch.sqrt(target_real ** 2 + target_imag ** 2 + 1e-8)

        pred_phase = torch.atan2(pred_imag, pred_real)
        target_phase = torch.atan2(target_imag, target_real)

        # phase cosine similarity loss
        phase_cosine = torch.cos(pred_phase - target_phase)
        phase_loss = 1 - phase_cosine.mean()

        return phase_loss

    def correlation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """correlation loss, keep structure of channel"""
        batch_size, seq_len, features = pred.shape

        # expand batch and vector dimensionality
        pred_flat = pred.view(batch_size * seq_len, features)
        target_flat = target.view(batch_size * seq_len, features)

        # calculate correlation coefficient matrix
        pred_corr = torch.corrcoef(pred_flat.T)
        target_corr = torch.corrcoef(target_flat.T)

        # verify validity of matrix
        pred_corr = torch.nan_to_num(pred_corr, nan=0.0)
        target_corr = torch.nan_to_num(target_corr, nan=0.0)

        # calculate Frobenius norm
        corr_loss = F.mse_loss(pred_corr, target_corr)

        return corr_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Total loss"""
        mse_loss = self.complex_mse_loss(pred, target)
        phase_loss = self.phase_cosine_loss(pred, target)
        corr_loss = self.correlation_loss(pred, target)

        total_loss = (self.mse_weight * mse_loss +
                      self.phase_weight * phase_loss +
                      self.correlation_weight * corr_loss)

        return total_loss, {
            'mse_loss': mse_loss.item(),
            'phase_loss': phase_loss.item(),
            'corr_loss': corr_loss.item(),
            'total_loss': total_loss.item()
        }
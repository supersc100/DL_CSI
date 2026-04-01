import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import Tuple


class CSIDataset(Dataset):
    """
    CSI数据集的PyTorch适配器
    从Sionna生成的HDF5文件加载数据
    """

    def __init__(self,
                 hdf5_path: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 normalize: bool = True):
        """
        Args:
            hdf5_path: Sionna生成的HDF5文件路径
            split: 数据分割 ('train', 'val', 'test')
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.normalize = normalize

        # 加载数据元信息
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f['downlink_real'].shape[0]
            self.antenna_shape = f['downlink_real'].shape[1:]  # (BS_ant, UE_ant, subcarriers)

        # 计算分割索引
        train_end = int(self.num_samples * train_ratio)
        val_end = train_end + int(self.num_samples * val_ratio)

        if split == 'train':
            self.indices = range(0, train_end)
        elif split == 'val':
            self.indices = range(train_end, val_end)
        else:  # test
            self.indices = range(val_end, self.num_samples)

        # 计算归一化参数（在训练集上计算，应用到所有集）
        self._compute_normalization_params()

        print(f"CSI数据集加载: {hdf5_path}")
        print(f"  分割: {split}, 样本数: {len(self.indices)}")
        print(f"  天线配置: {self.antenna_shape}")

    def _compute_normalization_params(self):
        """计算归一化参数（基于训练集）"""
        if not self.normalize:
            self.mean = 0.0
            self.std = 1.0
            return

        # 从训练集计算统计量
        train_indices = range(0, int(self.num_samples * 0.8))

        # 采样部分数据计算统计
        sample_size = min(1000, len(train_indices))
        sample_indices = np.random.choice(train_indices, sample_size, replace=False)

        real_values = []
        imag_values = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for idx in sample_indices:
                real = f['downlink_real'][idx]
                imag = f['downlink_imag'][idx]
                real_values.append(real.flatten())
                imag_values.append(imag.flatten())

        all_real = np.concatenate(real_values)
        all_imag = np.concatenate(imag_values)

        # 计算复合统计量
        self.mean_real = np.mean(all_real)
        self.std_real = np.std(all_real) + 1e-8
        self.mean_imag = np.mean(all_imag)
        self.std_imag = np.std(all_imag) + 1e-8

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 转换为全局索引
        global_idx = self.indices[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            # 读取CSI数据（实部和虚部分离存储）
            down_real = f['downlink_real'][global_idx]  # [BS_ant, UE_ant, subcarriers]
            down_imag = f['downlink_imag'][global_idx]
            up_real = f['uplink_real'][global_idx]
            up_imag = f['uplink_imag'][global_idx]

        # 归一化
        if self.normalize:
            down_real = (down_real - self.mean_real) / self.std_real
            down_imag = (down_imag - self.mean_imag) / self.std_imag
            up_real = (up_real - self.mean_real) / self.std_real
            up_imag = (up_imag - self.mean_imag) / self.std_imag

        # 合并为PyTorch张量
        # 形状: [BS_ant, UE_ant, subcarriers, 2] (最后一维是实部和虚部)
        downlink_csi = np.stack([down_real, down_imag], axis=-1)
        uplink_csi = np.stack([up_real, up_imag], axis=-1)

        # 转换为torch张量
        downlink_tensor = torch.FloatTensor(downlink_csi)
        uplink_tensor = torch.FloatTensor(uplink_csi)

        # 调整维度顺序以匹配您的模型输入
        # 从 [BS_ant, UE_ant, subcarriers, 2] 到 [seq_len, features]
        # 这里需要根据您的CSI编码器设计调整
        bs_ant, ue_ant, subcarriers, _ = downlink_tensor.shape
        downlink_reshaped = downlink_tensor.permute(2, 0, 1, 3).reshape(
            subcarriers, bs_ant * ue_ant * 2
        )
        uplink_reshaped = uplink_tensor.permute(2, 0, 1, 3).reshape(
            subcarriers, bs_ant * ue_ant * 2
        )

        return downlink_reshaped, uplink_reshaped

    def get_data_stats(self) -> dict:
        """获取数据统计信息"""
        return {
            'mean_real': self.mean_real,
            'std_real': self.std_real,
            'mean_imag': self.mean_imag,
            'std_imag': self.std_imag,
            'num_samples': len(self),
            'input_dim': self.antenna_shape[0] * self.antenna_shape[1] * self.antenna_shape[2] * 2
        }
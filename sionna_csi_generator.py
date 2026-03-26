import tensorflow as tf
import sionna
import numpy as np
from sionna.channel import *
from sionna.utils import *
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class SionnaCSIGenerator:
    """
    CSI Data generator based on Sionna
    generate DL CSI and corresponding UL CSI based on channel reciprocity
    """

    def __init__(self,
                 carrier_freq: float = 3.5e9,  # 3.5GHz (typical frequency band)
                 uplink_carrier_freq: float = None,  # 上行载波频率，若为None则视为TDD
                 subcarrier_spacing: float = 15e3,  # 15kHz
                 num_subcarriers: int = 64,  # submariners
                 num_antennas_bs: int = 4,  # antenna bs
                 num_antennas_ue: int = 2,  # antenna ue
                 bandwidth: float = 100e6,  # 100MHz bandwidth
                 scenario: str = "UMi",
                 seed: int = 42):  # 3GPP UMi
        # channel paras config
        self.carrier_freq = carrier_freq
        self.uplink_carrier_freq = uplink_carrier_freq if uplink_carrier_freq is not None else carrier_freq
        self.seed = seed
        self.subcarrier_spacing = subcarrier_spacing
        self.num_subcarriers = num_subcarriers
        self.num_antennas_bs = num_antennas_bs
        self.num_antennas_ue = num_antennas_ue
        self.bandwidth = bandwidth
        self.scenario = scenario

        # Sionna channel model init
        self._init_channel_model()

    def _init_channel_model(self):
        """3GPP channel model initialization"""
        # common paras（except freq）
        common_kwargs = {
            "model": self.scenario,
            "delay_spread": 100e-9,  # 可根据需要调整或作为参数传入
            "ue_speed": 3.0,  # UE移动速度
            "min_speed": 0.0
        }
        # generate CDL channel model (3GPP TR 38.901)
        self.cdl_down = CDL(carrier_frequency=self.carrier_freq,
                            seed=self.seed,
                            **common_kwargs)
        # 上行信道模型（使用相同种子保证路径几何一致）
        self.cdl_up = CDL(carrier_frequency=self.uplink_carrier_freq,
                          seed=self.seed,  # 相同种子确保随机相位、角度等一致
                          **common_kwargs)

        # config OFDM resource grid
        self.ofdm_resource_grid = sionna.ofdm.ResourceGrid(
            num_ofdm_symbols=14,  # symbol number in one slot
            fft_size=self.num_subcarriers,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=self.num_antennas_bs,
            num_streams_per_tx=self.num_antennas_ue,
            cyclic_prefix_length=6,  # CP length
            pilot_pattern="kronecker",  # pilot mode
            pilot_ofdm_symbol_indices=[2, 11]  # pilot symbols
        )

        # stream channel
        self.stream_manager = sionna.channel.StreamManagement()

        # channel estimation（to generate CSI）
        self.channel_estimator = sionna.ofdm.LMMSEEqualizer(
            self.ofdm_resource_grid,
            self.stream_manager
        )

        print(f"CSI generator init Done! - scenario: {self.scenario}")
        print(f"  antenna configuration: BS={self.num_antennas_bs}x UE={self.num_antennas_ue}")
        print(f"  carrier frequency: {self.carrier_freq / 1e9:.2f}GHz, bandWidth: {self.bandwidth / 1e6:.0f}MHz")

    def generate_channel_matrix(self, batch_size: int = 32) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        generate CSI matrix（DL and UL）

        Args:
            batch_size:

        Returns:
            h_downlink: DL CSI [batch, BS_ant, UE_ant, subcarriers]
            h_uplink: UL CSI [batch, UE_ant, BS_ant, subcarriers]
        """
        # generate DL channel impulse response
        h_taps_down, delays_down = self.cdl_down(batch_size=batch_size,
                                                 num_time_steps=1,
                                                 sampling_frequency=1 / self.subcarrier_spacing)
        # transform to frequency CSI
        # calculate frequency domain channel based on OFDM grid
        frequency_response = sionna.channel.generate_ofdm_channel(
            h_taps_down, delays_down,
            self.ofdm_resource_grid,
            normalize_channel=True
        )   # shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, subcarriers]

        # 生成上行信道脉冲响应
        h_taps_up, delays_up = self.cdl_up(batch_size=batch_size,
                                           num_time_steps=1,
                                           sampling_frequency=1 / self.subcarrier_spacing)
        freq_resp_up = sionna.channel.generate_ofdm_channel(
            h_taps_up, delays_up,
            self.ofdm_resource_grid,
            normalize_channel=True
        )
        # frequency_response shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, subcarriers]
        # assume user MIMO，
        batch_size = frequency_response.shape[0]

        # DLCSI
        h_downlink = tf.transpose(frequency_response[:, 0, :, 0, :, :],
                                  perm=[0, 3, 2, 1])
        # h_downlink: [batch, BS_ant, UE_ant, subcarriers]

        # UL CSI
        h_uplink = tf.transpose(frequency_response[:, 0, :, 0, :, :],
                                perm=[0, 2, 3, 1])
        # h_uplink: [batch, UE_ant, BS_ant, subcarriers]

        return h_downlink.numpy(), h_uplink.numpy()

    def generate_dataset(self,
                         num_samples: int = 10000,
                         batch_size: int = 100,
                         save_path: str = "csi_dataset.h5") -> Dict[str, np.ndarray]:
        """
        生成大规模CSI数据集并保存

        Args:
            num_samples: 总样本数
            batch_size: 每个批次的样本数
            save_path: 保存路径

        Returns:
            包含CSI数据的字典
        """
        print(f"开始生成 {num_samples} 个CSI样本...")

        # 初始化HDF5文件存储
        with h5py.File(save_path, 'w') as f:
            # 创建可扩展的数据集
            max_shape = (None, self.num_antennas_bs, self.num_antennas_ue, self.num_subcarriers)
            dset_down_real = f.create_dataset('downlink_real',
                                              shape=(0, *max_shape[1:]),
                                              maxshape=max_shape,
                                              dtype=np.float32,
                                              chunks=True,
                                              compression='gzip')
            dset_down_imag = f.create_dataset('downlink_imag',
                                              shape=(0, *max_shape[1:]),
                                              maxshape=max_shape,
                                              dtype=np.float32,
                                              chunks=True,
                                              compression='gzip')
            dset_up_real = f.create_dataset('uplink_real',
                                            shape=(0, *max_shape[1:]),
                                            maxshape=max_shape,
                                            dtype=np.float32,
                                            chunks=True,
                                            compression='gzip')
            dset_up_imag = f.create_dataset('uplink_imag',
                                            shape=(0, *max_shape[1:]),
                                            maxshape=max_shape,
                                            dtype=np.float32,
                                            chunks=True,
                                            compression='gzip')

            # 分批次生成数据
            num_batches = (num_samples + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(num_batches)):
                current_batch = min(batch_size, num_samples - batch_idx * batch_size)

                # 生成CSI数据
                h_down, h_up = self.generate_channel_matrix(current_batch)

                # 分离实部和虚部
                h_down_real = np.real(h_down).astype(np.float32)
                h_down_imag = np.imag(h_down).astype(np.float32)
                h_up_real = np.real(h_up).astype(np.float32)
                h_up_imag = np.imag(h_up).astype(np.float32)

                # 追加到HDF5文件
                new_size = dset_down_real.shape[0] + current_batch

                dset_down_real.resize((new_size, *max_shape[1:]))
                dset_down_imag.resize((new_size, *max_shape[1:]))
                dset_up_real.resize((new_size, *max_shape[1:]))
                dset_up_imag.resize((new_size, *max_shape[1:]))

                dset_down_real[-current_batch:] = h_down_real
                dset_down_imag[-current_batch:] = h_down_imag
                dset_up_real[-current_batch:] = h_up_real
                dset_up_imag[-current_batch:] = h_up_imag

        print(f"数据集已保存到: {save_path}")
        print(f"  下行CSI形状: ({num_samples}, {self.num_antennas_bs}, {self.num_antennas_ue}, {self.num_subcarriers})")

        # 加载并返回数据
        with h5py.File(save_path, 'r') as f:
            data = {
                'downlink': f['downlink_real'][:] + 1j * f['downlink_imag'][:],
                'uplink': f['uplink_real'][:] + 1j * f['uplink_imag'][:],
            }

        return data

    def visualize_channel(self, num_samples: int = 5):
        """可视化生成的CSI数据"""
        h_down, h_up = self.generate_channel_matrix(num_samples)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        for i in range(min(num_samples, 2)):
            # 下行CSI幅度
            ax = axes[i, 0]
            csi_amp = np.abs(h_down[i, 0, 0, :])  # 第一个BS天线到第一个UE天线
            ax.plot(csi_amp)
            ax.set_title(f'样本{i} - 下行CSI幅度')
            ax.set_xlabel('子载波索引')
            ax.set_ylabel('幅度')
            ax.grid(True, alpha=0.3)

            # 下行CSI相位
            ax = axes[i, 1]
            csi_phase = np.angle(h_down[i, 0, 0, :])
            ax.plot(csi_phase)
            ax.set_title(f'样本{i} - 下行CSI相位')
            ax.set_xlabel('子载波索引')
            ax.set_ylabel('相位 (rad)')
            ax.grid(True, alpha=0.3)

            # 上下行相关性
            ax = axes[i, 2]
            down_flat = h_down[i].flatten()
            up_flat = h_up[i].flatten()
            correlation = np.corrcoef([np.real(down_flat), np.imag(down_flat),
                                       np.real(up_flat), np.imag(up_flat)])

            im = ax.imshow(correlation[:2, 2:], cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'样本{i} - 上下行CSI相关性')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['上行实部', '上行虚部'])
            ax.set_yticklabels(['下行实部', '下行虚部'])
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig('csi_visualization.png', dpi=300)
        plt.show()
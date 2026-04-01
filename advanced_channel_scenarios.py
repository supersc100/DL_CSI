import numpy as np
import sionna_csi_generator
class AdvancedCSIGenerator:
    """
    高级CSI生成器，支持多种场景和信道效应
    """

    def __init__(self):
        self.scenarios = {
            "UMi": {"model": "UMi", "delay_spread": 100e-9, "ue_speed": 3.0},
            "UMa": {"model": "UMa", "delay_spread": 300e-9, "ue_speed": 5.0},
            "RMa": {"model": "RMa", "delay_spread": 50e-9, "ue_speed": 1.0},
            "Indoor": {"model": "InH", "delay_spread": 30e-9, "ue_speed": 0.5},
        }

    def generate_multi_scenario_dataset(self,
                                        samples_per_scenario: int = 2500,
                                        save_dir: str = "multi_scenario_csi"):
        """生成多场景CSI数据集"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        all_downlink = []
        all_uplink = []
        scenario_labels = []

        for scenario_name, params in self.scenarios.items():
            print(f"\n生成场景: {scenario_name}")

            # 创建对应场景的生成器
            generator = sionna_csi_generator.SionnaCSIGenerator(
                scenario=params["model"],
                delay_spread=params["delay_spread"],
                ue_speed=params["ue_speed"]
            )

            # 生成数据
            h_down, h_up = generator.generate_channel_matrix(samples_per_scenario)

            all_downlink.append(h_down)
            all_uplink.append(h_up)
            scenario_labels.extend([scenario_name] * samples_per_scenario)

        # 合并所有数据
        downlink_all = np.concatenate(all_downlink, axis=0)
        uplink_all = np.concatenate(all_uplink, axis=0)

        # 保存
        np.savez_compressed(
            f"{save_dir}/multi_scenario_csi.npz",
            downlink=downlink_all,
            uplink=uplink_all,
            scenarios=scenario_labels
        )

        print(f"\n多场景数据集已保存到: {save_dir}")
        print(f"总样本数: {len(downlink_all)}")
        print(f"场景分布: {dict(zip(*np.unique(scenario_labels, return_counts=True)))}")

        return downlink_all, uplink_all, scenario_labels

    def add_channel_impairments(self,
                                csi_data: np.ndarray,
                                noise_db: float = 20.0,
                                phase_noise_deg: float = 5.0,
                                iq_imbalance_db: float = 0.1) -> np.ndarray:
        """添加信道损伤（噪声、相位噪声、IQ不平衡）"""
        noisy_csi = csi_data.copy()

        # 1. 添加高斯噪声
        signal_power = np.mean(np.abs(csi_data) ** 2)
        noise_power = signal_power / (10 ** (noise_db / 10.0))
        noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*csi_data.shape) +
                1j * np.random.randn(*csi_data.shape)
        )
        noisy_csi += noise

        # 2. 添加相位噪声
        phase_noise = np.exp(1j * np.random.uniform(
            -np.deg2rad(phase_noise_deg),
            np.deg2rad(phase_noise_deg),
            csi_data.shape
        ))
        noisy_csi *= phase_noise

        # 3. 添加IQ不平衡
        iq_gain_imbalance = 10 ** (iq_imbalance_db / 20.0)
        iq_phase_imbalance = np.deg2rad(1.0)  # 1度相位不平衡

        # IQ不平衡模型
        i_real = np.real(noisy_csi)
        q_imag = np.imag(noisy_csi)

        i_imbalanced = i_real * iq_gain_imbalance
        q_imbalanced = q_imag * (1.0 / iq_gain_imbalance) * np.cos(iq_phase_imbalance)

        noisy_csi = i_imbalanced + 1j * q_imbalanced

        return noisy_csi
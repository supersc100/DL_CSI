import torch

import csi_deepseek_model
import csi_data_processor
import csi_loss_functions
import numpy as np
class CSIPredictor:
    """
    CSI预测器 - 部署和使用接口
    """

    def __init__(self, model_path: str = "csi_models/best_model.pt"):
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型架构
        self.model = csi_deepseek_model.CSIDeepSeekModel(
            model_name="deepseek-ai/deepseek-llm-7b-chat",
            csi_input_dim=128,
            csi_output_dim=128,
            use_lora=True,
            lora_r=8
        )

        # 加载训练好的权重
        self.model.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 初始化处理器
        self.processor = csi_data_processor.CSIDataProcessor()

        print(f"CSI预测器已加载，使用设备: {self.device}")

    def predict(self, downlink_csi: np.ndarray) -> np.ndarray:
        """
        预测上行CSI

        Args:
            downlink_csi: 下行CSI复数矩阵 [antennas, subcarriers]

        Returns:
            预测的上行CSI复数矩阵
        """
        # 预处理
        downlink_tensor = self.processor.preprocess_csi(downlink_csi[np.newaxis, ...])
        downlink_tensor = downlink_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            predicted_tensor = self.model(downlink_tensor)
            predicted_tensor = predicted_tensor.cpu().numpy()

        # 后处理：将预测结果转换回复数矩阵
        batch_size, seq_len, features = predicted_tensor.shape

        # 重构为原始形状
        predicted_reshaped = predicted_tensor.reshape(
            batch_size,
            self.processor.num_antennas,
            self.processor.num_subcarriers,
            2
        )

        # 转换为复数
        predicted_complex = predicted_reshaped[..., 0] + 1j * predicted_reshaped[..., 1]

        return predicted_complex[0]  # 返回第一个样本

    def evaluate_batch(self, downlink_batch: np.ndarray, uplink_batch: np.ndarray) -> dict:
        """
        批量评估模型性能

        Args:
            downlink_batch: 下行CSI批量数据 [batch, antennas, subcarriers]
            uplink_batch: 对应的真实上行CSI

        Returns:
            评估指标字典
        """
        # 预处理
        downlink_tensor = self.processor.preprocess_csi(downlink_batch)
        uplink_tensor = self.processor.preprocess_csi(uplink_batch)

        downlink_tensor = downlink_tensor.to(self.device)
        uplink_tensor = uplink_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(downlink_tensor)

            # 计算损失
            criterion = csi_loss_functions.CSILoss()
            loss, loss_dict = criterion(predictions, uplink_tensor)

            # 计算NMSE
            predictions_cpu = predictions.cpu().numpy()
            targets_cpu = uplink_tensor.cpu().numpy()

            mse = np.mean((predictions_cpu - targets_cpu) ** 2)
            power = np.mean(targets_cpu ** 2)
            nmse = mse / power

            # 计算相关系数
            pred_flat = predictions_cpu.reshape(-1)
            target_flat = targets_cpu.reshape(-1)
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]

        return {
            'loss': loss.item(),
            'nmse': nmse,
            'correlation': correlation,
            **loss_dict
        }


# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = CSIPredictor()

    # 生成测试数据
    processor = csi_data_processor.CSIDataProcessor()
    test_downlink, test_uplink = processor.generate_synthetic_data(num_samples=10)

    # 单样本预测
    print("\n单样本预测示例:")
    sample_idx = 0
    downlink_sample = test_downlink[sample_idx].numpy().reshape(4, 64)  # 假设为4x64矩阵
    predicted = predictor.predict(downlink_sample)
    print(f"预测形状: {predicted.shape}")

    # 批量评估
    print("\n批量评估:")
    metrics = predictor.evaluate_batch(
        test_downlink[:5].numpy().reshape(5, 4, 64),
        test_uplink[:5].numpy().reshape(5, 4, 64)
    )

    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
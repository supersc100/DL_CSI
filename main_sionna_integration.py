import torch

import sionna_csi_generator
import csi_data_loader
def main():
    """主程序：集成Sionna CSI生成与DeepSeek模型训练"""
    print("=" * 60)
    print("Sionna CSI数据生成与模型训练集成系统")
    print("=" * 60)

    # 步骤1: 生成CSI数据集
    print("\n[1/4] 生成CSI数据集...")
    generator = sionna_csi_generator.SionnaCSIGenerator(
        carrier_freq=3.5e9,
        uplink_carrier_freq=3.6e9,
        subcarrier_spacing=30e3,  # 30kHz子载波间隔
        num_subcarriers=64,
        num_antennas_bs=4,
        num_antennas_ue=2,
        bandwidth=100e6,
        scenario="UMi"
    )

    # 可视化示例
    generator.visualize_channel(num_samples=2)

    # 生成大规模数据集（根据需求调整大小）
    print("\n生成训练数据集...")
    data = generator.generate_dataset(
        num_samples=20000,  # 总样本数
        batch_size=200,
        save_path="csi_data/training_dataset.h5"
    )

    # 步骤2: 创建PyTorch数据加载器
    print("\n[2/4] 创建数据加载器...")
    from torch.utils.data import DataLoader

    train_dataset = csi_data_loader.CSIDataset(
        "csi_data/training_dataset.h5",
        split='train',
        train_ratio=0.7,
        val_ratio=0.15,
        normalize=True
    )

    val_dataset = csi_data_loader.CSIDataset(
        "csi_data/training_dataset.h5",
        split='val',
        train_ratio=0.7,
        val_ratio=0.15,
        normalize=True
    )

    # 获取数据统计信息，用于模型配置
    data_stats = train_dataset.get_data_stats()
    input_dim = data_stats['input_dim']
    print(f"CSI输入维度: {input_dim}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 步骤3: 初始化CSI-DeepSeek模型
    print("\n[3/4] 初始化CSI-DeepSeek模型...")
    from csi_deepseek_model import CSIDeepSeekModel

    model = CSIDeepSeekModel(
        model_name="deepseek-ai/deepseek-llm-7b-chat",
        csi_input_dim=input_dim,
        csi_output_dim=input_dim,  # 输入输出维度相同
        use_lora=True,
        lora_r=8
    )

    # 步骤4: 训练模型
    print("\n[4/4] 开始训练...")
    from training_pipeline import CSITrainingPipeline

    pipeline = CSITrainingPipeline(
        model=model,
        processor=None,  # 使用CSIDataset已经预处理了数据
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 训练
    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50
    )

    print("\n" + "=" * 60)
    print("训练完成! 模型已保存到 csi_models/ 目录")
    print("=" * 60)

    # 可选：生成测试集并评估
    print("\n生成测试数据集...")
    test_data = generator.generate_dataset(
        num_samples=2000,
        batch_size=100,
        save_path="csi_data/test_dataset.h5"
    )


if __name__ == "__main__":
    main()
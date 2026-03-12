def main():
    """主程序"""
    print("=" * 60)
    print("DeepSeek CSI预测系统启动")
    print("=" * 60)

    # 1. 初始化CSI处理器
    processor = CSIDataProcessor(
        seq_len=64,
        num_antennas=4,
        num_subcarriers=64
    )

    # 2. 生成/加载数据
    print("\n[1/4] 准备数据...")
    downlink_data, uplink_data = processor.generate_synthetic_data(num_samples=2000)

    # 划分训练集和验证集
    train_size = int(0.8 * len(downlink_data))
    train_downlink = downlink_data[:train_size]
    train_uplink = uplink_data[:train_size]
    val_downlink = downlink_data[train_size:]
    val_uplink = uplink_data[train_size:]

    print(f"  训练集: {len(train_downlink)} 样本")
    print(f"  验证集: {len(val_downlink)} 样本")

    # 3. 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_downlink, train_uplink)
    val_dataset = TensorDataset(val_downlink, val_uplink)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 4. 初始化模型
    print("\n[2/4] 初始化模型...")
    model = CSIDeepSeekModel(
        model_name="deepseek-ai/deepseek-llm-7b-chat",
        csi_input_dim=128,  # 根据processor.seq_len和特征数计算
        csi_output_dim=128,  # 输出维度应与输入相同
        use_lora=True,
        lora_r=8
    )

    # 5. 创建训练管道
    print("\n[3/4] 创建训练管道...")
    pipeline = CSITrainingPipeline(model, processor, device=device)

    # 6. 训练模型
    print("\n[4/4] 开始训练...")
    print("-" * 40)
    pipeline.train(train_loader, val_loader, epochs=30)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 7. 模型推理示例
    print("\n推理示例:")
    model.eval()
    with torch.no_grad():
        # 选择一个测试样本
        test_downlink = val_downlink[0:1].to(device)
        test_uplink = val_uplink[0:1].to(device)

        # 预测
        predicted_uplink = model(test_downlink)

        # 计算单个样本的NMSE
        pred_numpy = predicted_uplink.cpu().numpy()
        target_numpy = test_uplink.cpu().numpy()

        mse = np.mean((pred_numpy - target_numpy) ** 2)
        power = np.mean(target_numpy ** 2)
        nmse = mse / power

        print(f"  测试样本NMSE: {nmse:.6f}")

        # 可视化第一个特征
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(pred_numpy[0, :, 0], label='预测实部', alpha=0.7)
        plt.plot(target_numpy[0, :, 0], label='真实实部', alpha=0.7)
        plt.title('实部对比')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(pred_numpy[0, :, 64], label='预测虚部', alpha=0.7)
        plt.plot(target_numpy[0, :, 64], label='真实虚部', alpha=0.7)
        plt.title('虚部对比')
        plt.legend()

        plt.subplot(1, 3, 3)
        correlation = np.corrcoef(pred_numpy[0].flatten(), target_numpy[0].flatten())[0, 1]
        plt.scatter(pred_numpy[0].flatten(), target_numpy[0].flatten(), alpha=0.5, s=1)
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        plt.title(f'散点图 (r={correlation:.3f})')

        plt.tight_layout()
        plt.savefig('csi_models/inference_example.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
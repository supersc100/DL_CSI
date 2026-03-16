def main():
    """main"""
    print("=" * 60)
    print("DeepSeek CSI Prediction Start")
    print("=" * 60)

    # 1. initialize CSI processed初始化CSI处理器
    processor = CSIDataProcessor(
        seq_len=64,
        num_antennas=4,
        num_subcarriers=64
    )

    # 2. generate/load data
    print("\n[1/4] Preparing Data...")
    downlink_data, uplink_data = processor.generate_synthetic_data(num_samples=2000)

    # divide data into training set and verification set
    train_size = int(0.8 * len(downlink_data))
    train_downlink = downlink_data[:train_size]
    train_uplink = uplink_data[:train_size]
    val_downlink = downlink_data[train_size:]
    val_uplink = uplink_data[train_size:]

    print(f"  Training Set: {len(train_downlink)} Samples")
    print(f"  Verification Set: {len(val_downlink)} Samples")

    # 3. genetate data loader?
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_downlink, train_uplink)
    val_dataset = TensorDataset(val_downlink, val_uplink)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 4. initialize model
    print("\n[2/4] model initialization...")
    model = CSIDeepSeekModel(
        model_name="deepseek-ai/deepseek-llm-7b-chat",
        csi_input_dim=128,  # calculate based on processor.seq_len and feature number
        csi_output_dim=128,  # shape of output is same as that of input
        use_lora=True,
        lora_r=8
    )

    # 5. Construct training pipline
    print("\n[3/4]  Construct training pipline...")
    pipeline = CSITrainingPipeline(model, processor, device=device)

    # 6. training model
    print("\n[4/4] Start Training...")
    print("-" * 40)
    pipeline.train(train_loader, val_loader, epochs=30)

    print("\n" + "=" * 60)
    print("Traning Done!")
    print("=" * 60)

    # 7. model recognition sample
    print("\nrecognition sample:")
    model.eval()
    with torch.no_grad():
        # Select a testing sample
        test_downlink = val_downlink[0:1].to(device)
        test_uplink = val_uplink[0:1].to(device)

        # prediction
        predicted_uplink = model(test_downlink)

        # calculate NMSE for one sample
        pred_numpy = predicted_uplink.cpu().numpy()
        target_numpy = test_uplink.cpu().numpy()

        mse = np.mean((pred_numpy - target_numpy) ** 2)
        power = np.mean(target_numpy ** 2)
        nmse = mse / power

        print(f"  NMSE for testing sample: {nmse:.6f}")

        # virtualization for the first feature
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(pred_numpy[0, :, 0], label='prediction real part', alpha=0.7)
        plt.plot(target_numpy[0, :, 0], label='real part', alpha=0.7)
        plt.title('comparison for real part')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(pred_numpy[0, :, 64], label='prediction imag part', alpha=0.7)
        plt.plot(target_numpy[0, :, 64], label='imag part', alpha=0.7)
        plt.title('comparison for imag part')
        plt.legend()

        plt.subplot(1, 3, 3)
        correlation = np.corrcoef(pred_numpy[0].flatten(), target_numpy[0].flatten())[0, 1]
        plt.scatter(pred_numpy[0].flatten(), target_numpy[0].flatten(), alpha=0.5, s=1)
        plt.xlabel('prediction value')
        plt.ylabel('real value')
        plt.title(f'scatter diagram (r={correlation:.3f})')

        plt.tight_layout()
        plt.savefig('csi_models/inference_example.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
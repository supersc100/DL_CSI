class CSITrainingPipeline:
    """
    CSI预测训练管道
    """

    def __init__(self, model, processor, device='cuda'):
        self.model = model.to(device)
        self.processor = processor
        self.device = device

        # 自定义损失函数
        self.criterion = CSILoss(
            mse_weight=1.0,
            phase_weight=0.5,
            correlation_weight=0.3
        )

        # 仅优化可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (downlink, uplink) in enumerate(progress_bar):
            downlink, uplink = downlink.to(self.device), uplink.to(self.device)

            # 前向传播
            pred = self.model(downlink)

            # 计算损失
            loss, loss_dict = self.criterion(pred, uplink)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            self.optimizer.step()

            # 更新进度条
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{avg_loss:.6f}',
                'mse': f'{loss_dict["mse_loss"]:.6f}'
            })

        return total_loss / len(train_loader)

    def validate(self, val_loader) -> dict:
        """验证模型"""
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for downlink, uplink in val_loader:
                downlink, uplink = downlink.to(self.device), uplink.to(self.device)

                pred = self.model(downlink)
                loss, _ = self.criterion(pred, uplink)
                val_loss += loss.item()

                # 保存用于评估
                all_predictions.append(pred.cpu())
                all_targets.append(uplink.cpu())

        # 计算评估指标
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = self._compute_metrics(predictions, targets)
        metrics['val_loss'] = val_loss / len(val_loader)

        return metrics

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """计算评估指标"""
        batch_size, seq_len, features = pred.shape

        # 重构为复数
        pred_real = pred[:, :, :features // 2]
        pred_imag = pred[:, :, features // 2:]
        target_real = target[:, :, :features // 2]
        target_imag = target[:, :, features // 2:]

        # NMSE (归一化均方误差)
        mse = F.mse_loss(pred_real, target_real) + F.mse_loss(pred_imag, target_imag)
        power = torch.mean(target_real ** 2 + target_imag ** 2)
        nmse = mse / power

        # 相关系数
        pred_flat = pred.view(-1, features).numpy()
        target_flat = target.view(-1, features).numpy()

        # 计算平均相关系数
        corr_coeffs = []
        for i in range(min(10, features)):  # 只计算前10个特征的相关系数
            corr = np.corrcoef(pred_flat[:, i], target_flat[:, i])[0, 1]
            if not np.isnan(corr):
                corr_coeffs.append(corr)

        avg_correlation = np.mean(corr_coeffs) if corr_coeffs else 0

        return {
            'nmse': nmse.item(),
            'correlation': avg_correlation
        }

    def train(self, train_loader, val_loader, epochs: int = 50):
        """完整训练循环"""
        print("开始训练...")
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch + 1)
            self.train_losses.append(train_loss)

            # 验证
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['val_loss'])

            # 更新学习率
            self.scheduler.step()

            # 打印结果
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_metrics['val_loss']:.6f}")
            print(f"  NMSE: {val_metrics['nmse']:.6f}")
            print(f"  相关系数: {val_metrics['correlation']:.4f}")

            # 保存最佳模型
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.model.save_model(f"csi_models/best_model_epoch{epoch + 1}.pt")
                print(f"  保存最佳模型 (损失: {best_val_loss:.6f})")

            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                self.model.save_model(f"csi_models/checkpoint_epoch{epoch + 1}.pt")

        # 绘制训练曲线
        self._plot_training_curve()

    def _plot_training_curve(self):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', linewidth=2)
        plt.plot(self.val_losses, label='验证损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('CSI预测模型训练曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('csi_models/training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
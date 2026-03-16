class CSITrainingPipeline:
    """
    CSI training pipline
    """

    def __init__(self, model, processor, device='cuda'):
        self.model = model.to(device)
        self.processor = processor
        self.device = device

        # loss function
        self.criterion = CSILoss(
            mse_weight=1.0,
            phase_weight=0.5,
            correlation_weight=0.3
        )

        # optimize trainable para
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)

        # learn rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # log os training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, epoch: int) -> float:
        """train each epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (downlink, uplink) in enumerate(progress_bar):
            downlink, uplink = downlink.to(self.device), uplink.to(self.device)

            # propagation forward
            pred = self.model(downlink)

            # calculate loss
            loss, loss_dict = self.criterion(pred, uplink)

            # propagation backward
            self.optimizer.zero_grad()
            loss.backward()

            # clip grad
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            self.optimizer.step()

            # update process bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{avg_loss:.6f}',
                'mse': f'{loss_dict["mse_loss"]:.6f}'
            })

        return total_loss / len(train_loader)

    def validate(self, val_loader) -> dict:
        """model verification"""
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

                # save to evaluation
                all_predictions.append(pred.cpu())
                all_targets.append(uplink.cpu())

        # calculate evaluation index
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = self._compute_metrics(predictions, targets)
        metrics['val_loss'] = val_loss / len(val_loader)

        return metrics

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """calculate evaluation index"""
        batch_size, seq_len, features = pred.shape

        # re-constructed complex
        pred_real = pred[:, :, :features // 2]
        pred_imag = pred[:, :, features // 2:]
        target_real = target[:, :, :features // 2]
        target_imag = target[:, :, features // 2:]

        # NMSE (normal MSE)
        mse = F.mse_loss(pred_real, target_real) + F.mse_loss(pred_imag, target_imag)
        power = torch.mean(target_real ** 2 + target_imag ** 2)
        nmse = mse / power

        # correlation coeffs
        pred_flat = pred.view(-1, features).numpy()
        target_flat = target.view(-1, features).numpy()

        # calculate correlation coeffs
        corr_coeffs = []
        for i in range(min(10, features)):  # only calc corr coeffs for 10 features
            corr = np.corrcoef(pred_flat[:, i], target_flat[:, i])[0, 1]
            if not np.isnan(corr):
                corr_coeffs.append(corr)

        avg_correlation = np.mean(corr_coeffs) if corr_coeffs else 0

        return {
            'nmse': nmse.item(),
            'correlation': avg_correlation
        }

    def train(self, train_loader, val_loader, epochs: int = 50):
        """whole training circulation"""
        print("开始训练...")
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # training
            train_loss = self.train_epoch(train_loader, epoch + 1)
            self.train_losses.append(train_loss)

            # verification
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['val_loss'])

            # update learning rate更新学习率
            self.scheduler.step()

            # 打印结果
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  training loss: {train_loss:.6f}")
            print(f"  verification loss: {val_metrics['val_loss']:.6f}")
            print(f"  NMSE: {val_metrics['nmse']:.6f}")
            print(f"  corr coeffs: {val_metrics['correlation']:.4f}")

            # save optimal model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.model.save_model(f"csi_models/best_model_epoch{epoch + 1}.pt")
                print(f"  保存最佳模型 (损失: {best_val_loss:.6f})")

            # save check point per 10 epochs
            if (epoch + 1) % 10 == 0:
                self.model.save_model(f"csi_models/checkpoint_epoch{epoch + 1}.pt")

        # plot training curve
        self._plot_training_curve()

    def _plot_training_curve(self):
        """plot training and verification loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='training loss', linewidth=2)
        plt.plot(self.val_losses, label='verification loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('CSI prediction model training curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('csi_models/training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
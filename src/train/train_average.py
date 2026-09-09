"""Train the average-estimation network on Himawari or synthetic data."""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    masked_mse_loss,
)
from src.datasets.himawari_patch_dataset import create_himawari_loader
from src.models import AverageEstimationGenerator, Discriminator
from src.utils.config import load_config


class SyntheticAverageDataset(Dataset[dict[str, torch.Tensor]]):  # 学習処理: 形状確認用の平均推定サンプルを乱数で作る
    """Random tensors with the same shapes as average-estimation batches."""

    def __init__(
        self,
        *,
        num_samples: int,
        time_steps: int,
        patch_size: int,
        observed_probability: float = 0.8,
    ) -> None:
        super().__init__()
        assert num_samples > 0, f"num_samples must be positive, got {num_samples}"
        assert time_steps > 0, f"time_steps must be positive, got {time_steps}"
        assert patch_size > 0, f"patch_size must be positive, got {patch_size}"
        assert 0.0 < observed_probability <= 1.0, (
            "observed_probability must be in (0, 1], "
            f"got {observed_probability}"
        )
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.observed_probability = observed_probability

    def __len__(self) -> int:
        return self.num_samples

    def _mask(self, *shape: int) -> torch.Tensor:
        return (torch.rand(*shape) < self.observed_probability).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        height = width = self.patch_size
        time_steps = self.time_steps

        weekly_average = torch.rand(1, height, width) * 2.0 - 1.0
        monthly_average = weekly_average.unsqueeze(1) + 0.05 * torch.randn(
            1,
            1,
            height,
            width,
        )
        temporal_noise = 0.1 * torch.randn(1, time_steps, height, width)
        sst_volume = weekly_average.unsqueeze(1).repeat(1, time_steps, 1, 1)
        sst_volume = (sst_volume + temporal_noise).clamp(-1.0, 1.0)

        mask_volume = self._mask(1, time_steps, height, width)
        monthly_mask = self._mask(1, 1, height, width)
        weekly_mask = self._mask(1, height, width)
        assimilation_weekly = (weekly_average + 0.02 * torch.randn_like(weekly_average)).clamp(
            -1.0,
            1.0,
        )

        return {
            "sst_volume": sst_volume,
            "mask_volume": mask_volume,
            "monthly_average": monthly_average.clamp(-1.0, 1.0),
            "monthly_mask": monthly_mask,
            "weekly_average": weekly_average,
            "weekly_mask": weekly_mask,
            "assimilation_weekly": assimilation_weekly,
        }


def set_seed(seed: int) -> None:  # 学習処理: 再現性のためPython/NumPy/PyTorchの乱数を固定する
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:  # 学習処理: CUDA可否を見て実際に使うdeviceを決める
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:  # 学習処理: DataLoaderから来たバッチをdeviceへ移す
    return {key: value.to(device) for key, value in batch.items()}


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:  # 学習処理: Dの勾配を止める/戻す
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def create_synthetic_average_loader(
    config: Mapping[str, Any],
    *,
    num_samples: int,
) -> DataLoader[dict[str, torch.Tensor]]:  # 学習処理: 平均推定用の疑似バッチをDataLoader化する
    data_config = config["data"]
    training_config = config["training"]
    dataset = SyntheticAverageDataset(
        num_samples=num_samples,
        time_steps=int(data_config["time_steps"]),
        patch_size=int(data_config["patch_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(training_config.get("device") == "cuda" and torch.cuda.is_available()),
    )


def build_average_training_components(
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[  # 学習処理: 生成器・識別器・各最適化器を初期化する
    AverageEstimationGenerator,
    Discriminator,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    optimizer_config = config["optimizer"]
    generator = AverageEstimationGenerator(
        time_steps=int(config["data"]["time_steps"]),
    ).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(optimizer_config["lr_generator"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(optimizer_config["lr_discriminator"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
    )
    return generator, discriminator, optimizer_g, optimizer_d


def train_one_epoch(
    *,
    generator: AverageEstimationGenerator,
    discriminator: Discriminator,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: Mapping[str, Any],
    device: torch.device,
    epoch_index: int,
    max_batches: int | None = None,
) -> dict[str, float]:  # 学習処理: 生成器と識別器を交互に1エポック更新する
    generator.train()
    discriminator.train()

    loss_config = config["loss"]
    epochs = int(config["training"]["epochs"])
    lambda_rec = float(loss_config["lambda_ave_rec"])
    lambda_adv = float(loss_config["lambda_ave_adv"])
    epoch_weight = 1.0 - float(epoch_index) / max(float(epochs), 1.0)
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"

    def autocast_context():
        return torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()

    total_d = 0.0
    total_g = 0.0
    total_rec = 0.0
    total_adv = 0.0
    steps = 0

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = move_batch_to_device(raw_batch, device)
        monthly_average = batch["monthly_average"]
        monthly_mask = batch["monthly_mask"]
        weekly_average = batch["weekly_average"]
        weekly_mask = batch["weekly_mask"]
        with autocast_context():
            pred_weekly = generator(
                batch["sst_volume"],
                batch["mask_volume"],
                monthly_average,
                monthly_mask,
            )
        use_adversarial = "assimilation_weekly" in batch and lambda_adv > 0.0
        if use_adversarial:
            optimizer_d.zero_grad(set_to_none=True)
            with autocast_context():
                real_logits = discriminator(batch["assimilation_weekly"])
                fake_logits = discriminator(pred_weekly.detach())
                loss_d = discriminator_loss(real_logits, fake_logits)
            loss_d.backward()
            optimizer_d.step()
        else:
            loss_d = pred_weekly.new_zeros(())

        if use_adversarial:
            set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)
        with autocast_context():
            loss_reclong = masked_mse_loss(
                pred_weekly,
                monthly_average.squeeze(2),
                monthly_mask.squeeze(2),
            )
            loss_recshort = masked_mse_loss(pred_weekly, weekly_average, weekly_mask)
            loss_rec = epoch_weight * loss_reclong + loss_recshort
            if use_adversarial:
                fake_logits_for_g = discriminator(pred_weekly)
                loss_adv = generator_adversarial_loss(fake_logits_for_g)
            else:
                loss_adv = pred_weekly.new_zeros(())
            loss_g = lambda_rec * loss_rec + lambda_adv * loss_adv
        loss_g.backward()
        optimizer_g.step()
        if use_adversarial:
            set_requires_grad(discriminator, True)

        total_d += float(loss_d.detach().cpu())
        total_g += float(loss_g.detach().cpu())
        total_rec += float(loss_rec.detach().cpu())
        total_adv += float(loss_adv.detach().cpu())
        steps += 1

    assert steps > 0, "No training batches were processed."
    return {
        "loss_d": total_d / steps,
        "loss_g": total_g / steps,
        "loss_rec": total_rec / steps,
        "loss_adv": total_adv / steps,
        "epoch_weight": epoch_weight,
    }


def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> None:  # 学習処理: epochやoptimizer状態をcheckpointに保存する
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "config": dict(config),
        },
        path,
    )


def parse_args() -> argparse.Namespace:  # 学習処理: configやsynthetic指定のCLI引数を読む
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random tensors instead of preprocessed Himawari NPZ files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Directory containing NPZ files made by crop_nc_to_1650.py.",
    )
    parser.add_argument(
        "--synthetic-num-samples",
        type=int,
        default=None,
        help="Number of synthetic samples. Defaults to one epoch-sized batch set.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit on batches per epoch.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    parser.add_argument("--device", default=None, help="Override config device.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override config checkpoint directory.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:  # 学習処理: CLIで学習設定を上書きする
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.checkpoint_dir is not None:
        config["paths"]["checkpoint_dir"] = args.checkpoint_dir
    return config


def main() -> None:  # 学習処理: 実データまたはsyntheticで平均推定を学習する入口
    args = parse_args()
    if args.synthetic and args.data_root is not None:
        raise ValueError("Choose either --synthetic or --data-root, not both.")
    if not args.synthetic and args.data_root is None:
        raise ValueError("Real-data training requires --data-root.")

    config = apply_cli_overrides(load_config(args.config), args)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config["training"]["device"]))
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    if args.synthetic:
        synthetic_num_samples = args.synthetic_num_samples or batch_size * 2
        loader = create_synthetic_average_loader(
            config,
            num_samples=synthetic_num_samples,
        )
    else:
        assert args.data_root is not None
        loader = create_himawari_loader(args.data_root, config, stage="average")
        print(
            f"Loaded {len(loader.dataset)} Himawari samples from {args.data_root}. "
            "Data-assimilation fields are absent, so adversarial loss is disabled."
        )
    generator, discriminator, optimizer_g, optimizer_d = build_average_training_components(
        config,
        device,
    )

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"]) / "average"
    for epoch_index in range(epochs):
        metrics = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            config=config,
            device=device,
            epoch_index=epoch_index,
            max_batches=args.max_batches,
        )
        epoch = epoch_index + 1
        print(
            f"epoch={epoch}/{epochs} "
            f"loss_d={metrics['loss_d']:.6f} "
            f"loss_g={metrics['loss_g']:.6f} "
            f"loss_rec={metrics['loss_rec']:.6f} "
            f"loss_adv={metrics['loss_adv']:.6f}"
        )
        save_checkpoint(
            path=checkpoint_dir / "latest.pth",
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            config=config,
        )


if __name__ == "__main__":
    main()

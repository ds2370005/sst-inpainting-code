"""Run two-stage SST reconstruction from preprocessed array inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models import AnomalyInpaintingGenerator, AverageEstimationGenerator
from src.utils.config import load_config


REQUIRED_INPUT_KEYS = (
    "sst_volume",
    "mask_volume",
    "monthly_average",
    "monthly_mask",
)


def resolve_device(requested_device: str) -> torch.device:  # 推論処理: CUDAが使えればGPU、無理ならCPUを返す
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def _tensor_from_npz(archive: np.lib.npyio.NpzFile, key: str) -> torch.Tensor:  # 推論処理: npz内の指定キーをfloat32 Tensorに変換する
    assert key in archive, f"Input NPZ is missing required key {key!r}."
    return torch.as_tensor(np.asarray(archive[key]), dtype=torch.float32)


def _ensure_5d_volume(tensor: torch.Tensor, name: str) -> torch.Tensor:  # 推論処理: [B,1,T,H,W] になるよう次元を補う
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    assert tensor.ndim == 5, f"{name} must be [B, 1, T, H, W], got {tuple(tensor.shape)}"
    return tensor


def load_reconstruction_input(path: str | Path) -> dict[str, torch.Tensor]:  # 推論処理: 推論に必要なnpz入力をまとめて読む
    """Load inference inputs.

    The current implementation supports .npz files. Other formats can reuse the
    dataset loader adapter pattern when real-data integration is added.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise NotImplementedError("Inference currently supports .npz inputs only.")
    with np.load(path) as archive:
        arrays = {key: _tensor_from_npz(archive, key) for key in REQUIRED_INPUT_KEYS}
    return {key: _ensure_5d_volume(value, key) for key, value in arrays.items()}


def load_generator_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:  # 推論処理: 学習済み生成器のstate_dictをcheckpointから読む
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("generator_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def run_reconstruction(
    *,
    config: dict[str, Any],
    average_checkpoint: str | Path,
    anomaly_checkpoint: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    device: torch.device,
) -> dict[str, np.ndarray]:  # 推論処理: 平均推定と異常補完を順に通してSSTを再構成する
    inputs = {
        key: value.to(device)
        for key, value in load_reconstruction_input(input_path).items()
    }
    time_steps = int(config["data"]["time_steps"])
    average_generator = AverageEstimationGenerator(time_steps=time_steps).to(device)
    anomaly_generator = AnomalyInpaintingGenerator(time_steps=time_steps).to(device)
    load_generator_checkpoint(average_generator, average_checkpoint, device)
    load_generator_checkpoint(anomaly_generator, anomaly_checkpoint, device)
    average_generator.eval()
    anomaly_generator.eval()

    with torch.no_grad():
        pred_weekly = average_generator(
            inputs["sst_volume"],
            inputs["mask_volume"],
            inputs["monthly_average"],
            inputs["monthly_mask"],
        )
        pred_anomaly = anomaly_generator(
            inputs["sst_volume"],
            inputs["mask_volume"],
            pred_weekly.unsqueeze(2),
        )
        pred_sst = pred_weekly + pred_anomaly

    output = {
        "pred_weekly": pred_weekly.detach().cpu().numpy(),
        "pred_anomaly": pred_anomaly.detach().cpu().numpy(),
        "pred_sst": pred_sst.detach().cpu().numpy(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    return output


def parse_args() -> argparse.Namespace:  # 推論処理: 入力・出力・checkpointのCLI引数を読む
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--average-checkpoint", required=True)
    parser.add_argument("--anomaly-checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input .npz path.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--device", default=None, help="Override config device.")
    return parser.parse_args()


def main() -> None:  # 推論処理: CLIから再構成を実行する入口
    args = parse_args()
    config = load_config(args.config)
    requested_device = args.device or str(config["training"]["device"])
    device = resolve_device(requested_device)
    run_reconstruction(
        config=config,
        average_checkpoint=args.average_checkpoint,
        anomaly_checkpoint=args.anomaly_checkpoint,
        input_path=args.input,
        output_path=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exporta um checkpoint PyTorch de LPRNet para ONNX.

Uso:
  python scripts/export_lprnet_to_onnx.py \
      --model-module LPRNet_Pytorch.model.LPRNet \
      --model-class LPRNet \
      --weights LPRNet_Pytorch/weights/Final_LPRNet_model.pth \
      --output build/lprnet.onnx
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta um modelo LPRNet para ONNX")
    parser.add_argument("--model-module", required=True, help="Módulo python da classe do modelo")
    parser.add_argument("--model-class", required=True, help="Nome da classe do modelo")
    parser.add_argument("--weights", required=True, type=Path, help="Caminho para .pth")
    parser.add_argument("--output", required=True, type=Path, help="Saída .onnx")
    parser.add_argument("--img-h", type=int, default=24, help="Altura da entrada")
    parser.add_argument("--img-w", type=int, default=94, help="Largura da entrada")
    parser.add_argument("--channels", type=int, default=3, help="Quantidade de canais")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch de exportação")
    parser.add_argument("--opset", type=int, default=11, help="Opset ONNX")
    return parser.parse_args()


def load_model(module_name: str, class_name: str) -> torch.nn.Module:
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls()


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Formato de checkpoint não suportado")


def main() -> None:
    args = parse_args()

    model = load_model(args.model_module, args.model_class)
    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    # Remove prefixos comuns de DataParallel (module.)
    sanitized_state_dict = {
        k.replace("module.", "", 1): v for k, v in state_dict.items()
    }

    model.load_state_dict(sanitized_state_dict, strict=False)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(args.batch_size, args.channels, args.img_h, args.img_w)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )

    print(f"ONNX exportado com sucesso em: {args.output}")


if __name__ == "__main__":
    main()

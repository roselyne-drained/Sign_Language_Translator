from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class ASLModel:
    """Carga un modelo ASL y ofrece inferencia sobre frames de webcam."""

    def __init__(self, model_path: str | Path | None = None, labels: Iterable[str] | None = None, labels_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.labels = list(labels) if labels is not None else []
        self.labels_path = Path(labels_path) if labels_path else None
        self.backend: str | None = None
        self.session = None
        self.torch_model = None
        self.is_ready = False

        if self.labels_path and self.labels_path.exists():
            self._load_labels()

        if self.model_path and self.model_path.exists():
            self._load_model()

    def _load_labels(self) -> None:
        self.labels = [line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_model(self) -> None:
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError:  # pragma: no cover
                return

            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.backend = "onnx"
            self.is_ready = True
        elif suffix in {".pt", ".pth"}:
            try:
                import torch
            except ImportError:  # pragma: no cover
                return

            self.torch_model = torch.load(str(self.model_path), map_location="cpu")
            self.torch_model.eval()
            self.backend = "torch"
            self.is_ready = True
        else:
            self.is_ready = False

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return image

    def _decode(self, raw_output: np.ndarray) -> tuple[str, float]:
        raw_output = np.asarray(raw_output).squeeze()
        if raw_output.ndim == 0:
            index = int(raw_output)
            confidence = 1.0
        else:
            index = int(np.argmax(raw_output))
            confidence = float(np.max(raw_output))

        label = self.labels[index] if self.labels and 0 <= index < len(self.labels) else str(index)
        return label, confidence

    def predict(self, frame: np.ndarray) -> tuple[str, float]:
        if not self.is_ready:
            raise RuntimeError("No ASL model is loaded")

        inputs = self.preprocess(frame)
        if self.backend == "onnx":
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: inputs.astype(np.float32)})
            return self._decode(outputs[0])

        if self.backend == "torch":
            import torch

            with torch.no_grad():
                tensor = torch.from_numpy(inputs)
                output = self.torch_model(tensor)
                if hasattr(output, "numpy"):
                    raw = output.numpy()
                else:
                    raw = output.detach().cpu().numpy()
            return self._decode(raw)

        raise RuntimeError("Unsupported ASL model backend")

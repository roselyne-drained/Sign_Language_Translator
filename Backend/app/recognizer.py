"""Módulo inicial de reconocimiento de señas.

Este archivo contiene un reconocedor base para arrancar el proyecto.
La implementación actual es un placeholder heurístico para validar el
flujo en tiempo real (webcam -> backend -> texto).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import cv2
import numpy as np

from .model import ASLModel


@dataclass
class PredictionSmoother:
    """Suaviza predicciones por voto mayoritario en una ventana deslizante."""

    size: int = 8
    _history: Deque[str] = field(default_factory=deque)

    def push(self, label: str) -> str:
        self._history.append(label)
        if len(self._history) > self.size:
            self._history.popleft()

        votes = Counter(self._history)
        return votes.most_common(1)[0][0]


class SignRecognizer:
    """Reconocedor ASL con fallback para detección de movimiento."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._previous_gray: np.ndarray | None = None
        self._smoother = PredictionSmoother(size=8)
        self._last_label: str | None = None
        self._model = ASLModel(model_path=model_path)

    @property
    def model_loaded(self) -> bool:
        return self._model.is_ready

    def _motion_predict(self, frame_bgr: np.ndarray) -> tuple[str, float, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._previous_gray is None:
            self._previous_gray = gray
            return "SIN_SENA", 0.55, 0.0

        diff = cv2.absdiff(self._previous_gray, gray)
        motion_score = float(np.mean(diff))
        self._previous_gray = gray

        if motion_score < 2.0:
            return "SIN_SENA", 0.85, motion_score
        if motion_score < 8.0:
            return "MOVIMIENTO_BAJO", 0.75, motion_score
        return "MOVIMIENTO_ALTO", 0.7, motion_score

    def predict(self, frame_bgr: np.ndarray) -> dict[str, float | str]:
        if self._model.is_ready:
            try:
                label, confidence = self._model.predict(frame_bgr)
            except Exception:
                label, confidence, motion_score = self._motion_predict(frame_bgr)
                return {
                    "label": self._smoother.push(label),
                    "confidence": round(confidence, 2),
                    "motion_score": round(motion_score, 2),
                    "mode": "fallback",
                }

            return {
                "label": self._smoother.push(label),
                "confidence": round(confidence, 2),
                "mode": "model",
            }

        label, confidence, motion_score = self._motion_predict(frame_bgr)
        return {
            "label": self._smoother.push(label),
            "confidence": round(confidence, 2),
            "motion_score": round(motion_score, 2),
            "mode": "motion",
        }

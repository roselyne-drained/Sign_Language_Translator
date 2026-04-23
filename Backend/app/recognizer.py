"""Módulo inicial de reconocimiento de señas.

Este archivo contiene un reconocedor base para arrancar el proyecto.
La implementación actual es un placeholder heurístico para validar el
flujo en tiempo real (webcam -> backend -> texto).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque

import cv2
import numpy as np


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
    """Reconocedor MVP basado en movimiento.

    NOTA:
      - Esto NO es un modelo WLASL real todavía.
      - Solo clasifica actividad visual en tres estados para arrancar:
        `SIN_SENA`, `MOVIMIENTO_BAJO`, `MOVIMIENTO_ALTO`.
    """

    def __init__(self) -> None:
        self._previous_gray: np.ndarray | None = None
        self._smoother = PredictionSmoother(size=8)

    def predict(self, frame_bgr: np.ndarray) -> dict[str, float | str]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._previous_gray is None:
            self._previous_gray = gray
            return {"label": "SIN_SENA", "confidence": 0.55}

        diff = cv2.absdiff(self._previous_gray, gray)
        motion_score = float(np.mean(diff))
        self._previous_gray = gray

        if motion_score < 2.0:
            raw_label = "SIN_SENA"
            confidence = 0.85
        elif motion_score < 8.0:
            raw_label = "MOVIMIENTO_BAJO"
            confidence = 0.75
        else:
            raw_label = "MOVIMIENTO_ALTO"
            confidence = 0.7

        smoothed = self._smoother.push(raw_label)
        return {
            "label": smoothed,
            "confidence": round(confidence, 2),
            "motion_score": round(motion_score, 2),
        }
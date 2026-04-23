from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .recognizer import SignRecognizer

app = FastAPI(title="Sign Meet API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.environ.get("ASL_MODEL_PATH")
if not model_path:
    default_model = Path(__file__).resolve().parent / "models" / "asl.onnx"
    model_path = str(default_model) if default_model.exists() else None

recognizer = SignRecognizer(model_path=model_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sign-meet-backend"}


@app.get("/status")
def status() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": recognizer.model_loaded,
        "model_path": model_path or "",
        "mode": "backend",
    }


def decode_base64_image(data_url: str) -> np.ndarray:
    """Decodifica una imagen en formato data URL a matriz BGR de OpenCV."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    binary = base64.b64decode(encoded)
    array = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen")

    return image


@app.websocket("/ws/infer")
async def infer_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            frame_data = payload.get("frame")

            if not frame_data:
                await websocket.send_json(
                    {"error": "Payload inválido. Se esperaba el campo 'frame'."}
                )
                continue

            try:
                frame = decode_base64_image(frame_data)
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            prediction = recognizer.predict(frame)
            await websocket.send_json({"type": "prediction", "result": prediction})

    except WebSocketDisconnect:
        return

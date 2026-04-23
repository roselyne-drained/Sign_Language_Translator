# Sign Language Translator

Proyecto MVP de traducción de ASL con webcam y backend FastAPI.

## Estructura

- `Backend/`: servidor FastAPI y lógica de reconocimiento
- `Frontend/`: página web que captura la webcam y envía frames por WebSocket

## Cómo correr

1. Activa el entorno Python 3.11:
   ```powershell
   .\.venv311\Scripts\Activate.ps1
   ```

2. Instala dependencias:
   ```powershell
   pip install -r Backend/requirements.txt
   ```

3. Inicia el backend:
   ```powershell
   cd Backend
   uvicorn app.main:app --reload
   ```

4. Inicia el frontend:
   ```powershell
   cd ..\Frontend
   python -m http.server 3000 --directory .
   ```

5. Abre en el navegador:
   - `http://127.0.0.1:3000/index.html`
   - `http://127.0.0.1:8000/health`

## Cómo usar WLASL

Para avanzar a traducción con WLASL necesitas un modelo entrenado.

### 1. Entrena un modelo WLASL

- Usa `WLASL` para entrenar un modelo de reconocimiento de signos
- Exporta el modelo a ONNX o PyTorch

### 2. Coloca el modelo en el backend

- `Backend/app/models/asl.onnx`
- o establece la variable de entorno:
  ```powershell
  $env:ASL_MODEL_PATH = "C:\ruta\a\asl.onnx"
  ```

### 3. Reinicia el backend

- El backend detectará el modelo y usará inferencia real si está disponible

## Qué hace ahora

- El frontend construye una "frase" a partir de las etiquetas reconocidas
- El backend usa un `ASLModel` si se encuentra un modelo ONNX/PyTorch
- Si no hay modelo, sigue usando un fallback de detección de movimiento

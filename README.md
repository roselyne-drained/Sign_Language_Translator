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

   > Si aún no tienes el entorno, créalo con Python 3.11:
   > `python -m venv .venv311`

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
- `Backend/app/models/labels.txt`
- o establece la variable de entorno:
  ```powershell
  $env:ASL_MODEL_PATH = "C:\ruta\a\asl.onnx"
  $env:ASL_LABELS_PATH = "C:\ruta\a\labels.txt"
  ```

### 3. Reinicia el backend

- El backend detectará el modelo y usará inferencia real si está disponible

## Preparar WLASL para entrenamiento

1. Descarga el dataset WLASL desde el repositorio oficial:
   - https://github.com/dxli94/WLASL
   - sigue las instrucciones para obtener los videos y los metadatos.

2. Genera un manifiesto para entrenamiento:
   ```powershell
   py "Backend/scripts/prepare_wlasl.py" --root C:\ruta\a\WLASL --output Backend/data
   ```

3. Revisa los archivos generados en `Backend/data`:
   - `labels.txt`
   - `train.csv`
   - `val.csv`
   - `test.csv`

4. Entrena un modelo de clasificación de signos usando los archivos generados.

## Qué hace ahora

- El frontend construye una "frase" a partir de las etiquetas reconocidas
- El backend usa un `ASLModel` si se encuentra un modelo ONNX/PyTorch
- Si no hay modelo, sigue usando un fallback de detección de movimiento

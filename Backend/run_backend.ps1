$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
$venv = Join-Path $root '..\.venv311\Scripts\Activate.ps1'
if (-Not (Test-Path $venv)) {
    Write-Error "No se encontró .venv311 en la raíz del proyecto. Crea el entorno con Python 3.11 y ejecuta de nuevo."
    exit 1
}
& $venv
uvicorn app.main:app --reload

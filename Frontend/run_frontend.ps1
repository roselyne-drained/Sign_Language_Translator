$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
python -m http.server 3000 --directory .

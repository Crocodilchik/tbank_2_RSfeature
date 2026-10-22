$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path .cache,.tmp,.python | Out-Null
. "$PSScriptRoot\project_env.ps1"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Do not install it globally: request an approved local installer.'
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv --no-config python install 3.12.14 --install-dir "$PSScriptRoot\.python" --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Local Python installation failed' }
    $LocalInterpreter=Get-ChildItem -LiteralPath '.python' -Directory -Filter 'cpython-*' | ForEach-Object { Join-Path $_.FullName 'python.exe' } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $LocalInterpreter) { throw 'Local interpreter not found' }
    uv --no-config venv .venv --python $LocalInterpreter
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
uv --no-config pip install --python .venv\Scripts\python.exe 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Local CUDA PyTorch installation failed' }
uv --no-config pip install --python .venv\Scripts\python.exe -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Locked dependencies installation failed' }
Write-Output 'Local environment is ready. Use ./run.ps1 -Mode test'

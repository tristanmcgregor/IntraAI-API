param(
	[switch]$NoInstall,
	[switch]$RebuildVenv
)

$ErrorActionPreference = "Stop"

# Move to the backend directory (script location)
Set-Location -Path $PSScriptRoot

# Rebuild venv if requested
if ($RebuildVenv -and (Test-Path .\.venv)) {
	Write-Host "[start] Removing existing venv..."
	Remove-Item -Recurse -Force .\.venv
}

# Ensure virtual environment exists
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
	Write-Host "[start] Creating virtual environment..."
	python -m venv .venv
}

# Activate venv
Write-Host "[start] Activating venv..."
. .\.venv\Scripts\Activate.ps1

# Install requirements unless skipped
if (-not $NoInstall) {
	Write-Host "[start] Installing requirements..."
	python -m pip install -U pip
	python -m pip install -r requirements.txt
}

Write-Host "[start] Running API at http://127.0.0.1:8080"
python -m uvicorn app.main:app --reload --port 8080 
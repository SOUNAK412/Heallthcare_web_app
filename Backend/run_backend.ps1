# Run backend (PowerShell helper)
Set-Location -Path $PSScriptRoot
Write-Output "Installing Python dependencies (may ask for permission)..."
python -m pip install -r requirements.txt
Write-Output "Generating svc.pkl (dummy model) if missing..."
python generate_model_from_module.py
Write-Output "Starting Flask app (main.py)..."
python main.py

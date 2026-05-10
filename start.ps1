# Navigate to the script's directory
Set-Location $PSScriptRoot

# Activate the virtual environment
# Note: We use & to execute the script path
& .\.venv\Scripts\Activate.ps1

# The prompt is now modified to show (.venv)
# You can add your python command here, e.g.:
# python main.py
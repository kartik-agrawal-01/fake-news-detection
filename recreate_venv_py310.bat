@echo off
echo === Recreating Venv for Python 3.10 ===
echo Step 1: Deleting existing venv...
rmdir /s /q venv
if errorlevel 1 (
    echo Venv not found or deletion failed - continuing.
)

echo Step 2: Creating new venv with Python 3.10...
py -3.10 -m venv venv
if errorlevel 1 (
    echo Error: Python 3.10 not found. Install from python.org.
    pause
    exit /b 1
)

echo Step 3: Activating venv...
call venv\Scripts\activate.bat

echo Step 4: Upgrading pip...
python -m pip install --upgrade pip

echo Step 5: Installing torch CPU first (for 3.10 compatibility)...
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

echo Step 6: Installing other requirements...
pip install -r requirements.txt

echo Step 7: Testing Python 3.10 in venv...
python -c "import sys; print('Python Version:', sys.version)"

echo Step 8: Testing key imports...
python -c "import pandas as pd; print('Pandas OK:', pd.__version__)"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
python -c "import torch; print('Torch OK:', torch.__version__)"

echo === Done! Venv ready for Python 3.10. Run 'python fine_tune_model.py' next. ===
echo Deactivate with 'deactivate' when finished.
pause
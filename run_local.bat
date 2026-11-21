@echo off
REM DeepTrace - run the app locally on Windows (backup for the presentation)
cd /d %~dp0
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps facenet-pytorch
python -m pip install streamlit opencv-python-headless pandas pillow requests
python -m streamlit run app.py
pause

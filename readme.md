
# Setup steps:
### 1. Open/create virtual environment using python 3.10.11 in pycharm.
### 2. Activate the virtual environment using the pycharm terminal.
### 3. Run the command below in pycharm terminal.
```
python -m pip install -r requirements.txt
```
### 4. Install Tesseract Model from:
```
https://github.com/UB-Mannheim/tesseract/wiki
```
### 5. Set the path of Tesseract Model installed in system variable.
**Change this depending on where you saved the installation files.**
```
C:\Program Files\Tesseract-OCR
```
### 6. While in the project directory (StudentIDDetection), run the code by entering the following command into the terminal:
```commandline
python GUI.py
```
***
## Error Handler
### If you found the error below:
ERROR: Failed building wheel for detectron2\
  Running setup.py clean for detectron2\
Failed to build detectron2\
ERROR: Could not build wheels for detectron2, which is required to install pyproject.toml-based projects


### Copy the code below and run it in terminal, wait until it finish indexing
```markdown
pip install git+https://github.com/facebookresearch/detectron2.git 

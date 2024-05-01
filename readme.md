
# Setup steps:
### 1. Open/create virtual environment using python 3.10.11 in pycharm
### 2. Run the command below in pycharm terminal.
```
python -m pip install -r requirements.txt
```
### 3. Install Tesseract Model from
```
https://github.com/UB-Mannheim/tesseract/wiki
```
### 4. Set the path of Tesseract Model installed in system variable
**Change this depending on where you saved**
```
C:\Program Files\Tesseract-OCR
```
### 5. Done, you should be able to run the completed code without any issues.
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

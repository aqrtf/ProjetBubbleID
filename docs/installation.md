(La procedure d'installation est reprise de  https://github.com/cldunlap73/BubbleID)

First download and install the latest [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)


Install [Python](https://www.python.org/downloads/)

Create a new environment with python 3.10, we used [anaconda](https://www.anaconda.com/download)

```bash
conda create -n bubbleid python==3.10
conda activate bubbleid
```

Install [git](https://git-scm.com/install/)

Update dependencies:
```bash
pip install --upgrade pip setuptools wheel
```
Install [pytorch](https://pytorch.org/get-started/locally/) (Make sure cuda version matches if using gpu):
```bash
# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Install [detectron2](https://github.com/facebookresearch/detectron2):
```bash
pip install git+https://github.com/facebookresearch/detectron2
```
Install Additional Dependencies:
```bash
pip install numpy==1.23 opencv-python filterpy super-gradients
```
Git clone the repository
```bash
git clone https://github.com/aqrtf/ProjetBubbleID.git
```

All the code was run on Windows 11. A warning like this one will appear but doesn't impact the performance of the code
```bash
[2025-11-23 10:49:13] INFO - crash_tips_setup.py - Crash tips is enabled. You can set your environment variable to CRASH_HANDLER=FALSE to disable it
W1123 10:49:14.237000 2428 site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
[2025-11-23 10:49:26] WARNING - env_sanity_check.py - Failed to verify operating system: Deci officially supports only Linux kernels. Some features may not work as expected.
```

Si VSC est utilisé il faut mettre le Python interpreter sur bubbleid:
Ctrl+Shift+P
Python:Select interpreter


# Industrial-Scale Computer Vision Pipeline

This repository demonstrates an **industry-ready scalable ML pipeline** for 6D object pose estimation and robotic perception.

### Key Features
- Scalable dataset with **lazy loading** (HDF5)
- On-the-fly augmentations for training
- Modular code (dataset, models, training, inference)
- Reproducible configs (YAML)
- Experiment tracking (folder-based)

### Tech Stack
Python | PyTorch | OpenCV | HDF5 | YAML

### Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Prepare HDF5 dataset (from data/raw and data/splits):
```bash
python src/data/hdf5_writer.py
```
3. Train:
```bash
python src/training/train.py --config configs/train.yaml
```
4. Inference:
```bash
python src/inference/predict.py --model path/to/checkpoint.pth
```

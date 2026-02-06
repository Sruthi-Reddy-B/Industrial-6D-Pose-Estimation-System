
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
2. Prepare your data:
   - Place raw images in `data/raw/train/`, `data/raw/val/`, `data/raw/test/`
   - Update split files in `data/splits/` with your image paths and labels

3. Build HDF5 dataset:
```bash
python src/data/hdf5_writer.py
```
3. Train:
```bash
python src/training/train.py
```

The model will save to `experiments/exp_001/model.pth`

4. Inference

```bash
python src/inference/predict.py --model experiments/exp_001/model.pth --image path/to/image.jpg
```

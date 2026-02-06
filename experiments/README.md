# Experiments

Each subfolder here represents one experiment run.

## Structure
```
experiments/
├── exp_001/
│   ├── model.pth          # Trained model checkpoint
│   ├── config.yaml        # Config used for this run
│   └── training_log.txt   # Training metrics
├── exp_002/
└── ...
```

## Usage
- Experiments are automatically created when you run training
- Each experiment preserves the exact config used
- Easy to compare different hyperparameters or architectures

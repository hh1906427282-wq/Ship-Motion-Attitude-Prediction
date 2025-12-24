# Ship-Motion-Attitude-Prediction
A Multi-Scale and Temporal Dependency-Aware Deep Learning Framework for Multi-Task Ship Motion Attitude Prediction

# Multi-Scale CNN–MI-BiLSTM–Transformer with FusionGate  
## Multi-task Ship Motion & Attitude Prediction (6DoF)

This repository provides an end-to-end deep learning framework for short-term ship motion and attitude prediction.  
The model jointly predicts **position, velocity, and acceleration** of 6-DoF ship motions using a multi-scale, time-dependent perception architecture.

---

## 1. File Structure

```

├── data/
│ ├── dataset.py # Dataset loading, normalization, sliding window construction
│ ├── train.csv # Training dataset (normalized after processing)
│ ├── test.csv # Test dataset (normalized after processing)
│ └── 五级海况.csv # Original raw data (sea state level 5)
│
├── model/
│ ├── cnn_bilstm.py # Multi-scale CNN + MI-BiLSTM branch
│ ├── transformer.py # Transformer encoder branch
│ ├── fusion.py # FusionGate for dual-branch feature fusion
│ ├── multi_head.py # Multi-task output heads (pos / vel / acc)
│ ├── predictor.py # Complete prediction model
│ ├── train.py # Training script
│ └── test.py # Testing and evaluation script
│
└── README.md

````

---

## 2. Environment Requirements

- Python ≥ 3.8  
- PyTorch  
- NumPy  
- Pandas  
- scikit-learn  
- Matplotlib  
- Seaborn  
- Joblib  

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn joblib
````

---

## 3. Dataset Format

Input data must be provided as a **CSV file** with **18 columns**, ordered as:

```
x, y, z, rx, ry, rz,
xv, yv, zv, rxv, ryv, rzv,
xa, ya, za, rxa, rya, rza
```

Each row corresponds to one time step:

* Position (6 DoF)
* Velocity (6 DoF)
* Acceleration (6 DoF)

---

## 4. Data Processing

Implemented in `dataset.py`:

* Min–Max normalization using `MinMaxScaler`
* Sliding-window construction:

  * Input length: `input_len = 300`
  * Prediction length: `pred_len = 100`
* Dataset split:

  * Last **20,000** time steps are used for testing
* Saved files:

  * `test_input_data.npy`
  * `scaler_cnn_bilstm_attention.save`

---

## 5. Model Architecture

### 5.1 CNN–MI-BiLSTM Branch

* Multi-scale 1D CNN (kernel sizes: 3 / 5 / 7)
* Channel concatenation with residual shortcut
* MI-BiLSTM with learnable input gate

### 5.2 Transformer Branch

* Linear embedding of 18-D input
* Positional encoding
* Transformer encoder layers
* Temporal pooling to match prediction horizon

### 5.3 FusionGate

Adaptive fusion of dual-branch features:

```
gate = sigmoid(W [f_cnn; f_trans] + b)
fused = gate ⊙ f_cnn + (1 − gate) ⊙ f_trans
```

### 5.4 Multi-task Output Heads

Three independent MLP heads:

* Position: 6 DoF
* Velocity: 6 DoF
* Acceleration: 6 DoF

Final output:

```
[B, pred_len, 18]
```

---

## 6. Training

### Hyperparameters

* Input length: 300
* Prediction length: 100
* Batch size: 32
* Epochs: 300
* Optimizer: Adam
* Learning rate: 1e-3
* Early stopping patience: 15

### Loss Function

The total loss is defined as:

```
L = L_pos + L_vel + L_acc
```

Each term is computed using Mean Squared Error (MSE).

### Run Training

```bash
python train.py
```

Outputs:

* Best model: `checkpoints/best_model.pth`
* Training curve: `loss_figs/train_loss_curve.png`
* Normalization scaler file

> ⚠️ Update the CSV path in `train.py` before running.

---

## 7. Testing and Evaluation

Implemented in `test.py`:

* Sliding-window inference with step size = `pred_len`
* Inverse normalization before evaluation
* Metrics computed **per channel (18 dimensions)**

### Metrics

For each channel (k):

* RMSE
* MAE
* MAPE
* R²

### Run Testing

```bash
python test.py
```

Outputs:

* Prediction CSV:

  ```
  cnn_bilstm_attention_multitask_prediction_testset.csv
  ```
* FusionGate weights:

  ```
  fusion_attention_weights.csv
  fusion_attention_heatmap.png
  ```
* Prediction figures:

  ```
  figures/*.png
  ```

---

## 8. Reproducibility

To ensure reproducibility:

1. Fixed input/output window lengths (300 / 100)
2. Deterministic train–test split
3. Saved scaler and normalized test input
4. Explicit model checkpointing
5. Evaluation strictly follows `test.py`

---

## 9. Citation

If you use this code in your research, please cite:

```bibtex
@article{ShipMotionMultiTask2025,
  title   = {Multi-scale and Time-dependent Perception Framework for Multi-task Ship Motion Prediction},
  author  = {Author Names},
  journal = {Applied Ocean Research},
  year    = {2025}
}
```

---

## 10. License

This repository is provided for **research and academic use only**.

```


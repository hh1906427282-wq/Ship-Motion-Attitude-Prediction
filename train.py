import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.predictor import Predictor
from data.dataset import load_data
import matplotlib.pyplot as plt
import joblib
import numpy as np

# ---------- 训练参数 ----------
input_len = 300
pred_len = 100
input_dim = 18
batch_size = 32
epochs = 300
lr = 0.001
patience = 15
min_delta = 1e-4

# ---------- 准备设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 加载数据 ----------
train_dataset, test_dataset, scaler = load_data(
    csv_path='../多尺度 CNN + MI-BiLSTM + transformer + MLP 输出（去除attention注意力融合）/五级海况.csv',
    input_len=input_len,
    pred_len=pred_len,
    normalize=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------- 初始化模型 ----------
model = Predictor(input_dim=input_dim, pred_len=pred_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- EarlyStopping ----------
best_loss = float('inf')
trigger_count = 0
train_loss_list = []

# ---------- 训练循环 ----------
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred, _ = model(x)  # ✅ 只取预测结果，不用注意力
        loss_x = criterion(y_pred[:, :, 0:6],  y[:, :, 0:6])    # 位置
        loss_v = criterion(y_pred[:, :, 6:12], y[:, :, 6:12])   # 速度
        loss_a = criterion(y_pred[:, :, 12:18], y[:, :, 12:18]) # 加速度
        loss = loss_x + loss_v + loss_a

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | x: {loss_x.item():.4f} | v: {loss_v.item():.4f} | a: {loss_a.item():.4f}")

    # ---------- EarlyStopping 判断 ----------
    if avg_loss + min_delta < best_loss:
        best_loss = avg_loss
        trigger_count = 0
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print("✅ Best model saved.")
    else:
        trigger_count += 1
        print(f"↪️ 无提升，EarlyStopping 计数：{trigger_count}/{patience}")
        if trigger_count >= patience:
            print("🛑 触发 EarlyStopping，训练提前结束")
            break

# ---------- 保存损失图 ----------
plt.figure(figsize=(8, 4))
plt.plot(train_loss_list, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()
os.makedirs("loss_figs", exist_ok=True)
plt.savefig("loss_figs/train_loss_curve.png", dpi=300)
print("📉 训练损失曲线已保存：loss_figs/train_loss_curve.png")

# ---------- 保存 scaler 和测试数据 ----------
joblib.dump(scaler, 'scaler_cnn_bilstm_attention.save')
np.save("test_input_data.npy", test_dataset.data)
print("✅ scaler 保存为 scaler_cnn_bilstm_attention.save")
print("✅ 测试数据保存为 test_input_data.npy")

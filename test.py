import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import seaborn as sns
from model.predictor import Predictor

# ---------- 参数 ----------
input_len = 300
pred_len = 100
input_dim = 18
labels = ['x', 'y', 'z', 'rx', 'ry', 'rz']
groups = ['pos', 'vel', 'acc']  # 对应：位置、速度、加速度

# ---------- 加载 scaler 和测试数据 ----------
scaler: MinMaxScaler = joblib.load("scaler_cnn_bilstm_attention.save")
test_data = np.load("test_input_data.npy")  # [T, 18]
total_len = len(test_data)
print(f"✅ 加载测试数据，共 {total_len} 帧 ≈ {total_len * 0.1:.1f} 秒")

# ---------- 加载模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Predictor(input_dim=input_dim, pred_len=pred_len).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()
print("✅ 模型加载完成")

# ---------- 滑窗连续预测 ----------
predictions = []
ground_truths = []
attention_saved = False  # 只保存第一个样本的注意力图

start_indices = list(range(0, total_len - input_len - pred_len + 1, pred_len))  # 每100帧滑窗

for step_idx, start_idx in enumerate(start_indices):
    x_input = test_data[start_idx:start_idx + input_len]
    y_true = test_data[start_idx + input_len:start_idx + input_len + pred_len]

    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred_tensor, attn_weights = model(x_tensor)

    y_pred = y_pred_tensor.cpu().numpy().squeeze()  # [100, 18]

    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_pad = np.zeros_like(y_true)
    y_pred_pad[:, :18] = y_pred
    y_pred_inv = scaler.inverse_transform(y_pred_pad)

    ground_truths.append(y_true_inv)
    predictions.append(y_pred_inv)

    # ---------- 注意力可视化：保存第一个样本 ----------
    if not attention_saved:
        attn_map = attn_weights.squeeze(0).cpu().numpy()  # [T=300, D=128]
        np.savetxt("fusion_attention_weights.csv", attn_map, delimiter=",")
        print("✅ 融合注意力权重已保存：fusion_attention_weights.csv")

        # 绘图
        plt.figure(figsize=(12, 6))
        sns.heatmap(attn_map.T, cmap='viridis', cbar=True)
        plt.xlabel("Time step (0~299)")
        plt.ylabel("Feature dim (0~127)")
        plt.title("Fusion Attention Heatmap (BiLSTM vs Transformer)")
        plt.tight_layout()
        plt.savefig("fusion_attention_heatmap.png", dpi=300)
        plt.close()
        print("✅ 注意力热图已保存：fusion_attention_heatmap.png")

        attention_saved = True

# ---------- 拼接结果 ----------
y_true_full = np.vstack(ground_truths)
y_pred_full = np.vstack(predictions)
frame_rate = 10  # 10Hz
total_pred_len = y_true_full.shape[0]
time_axis = np.linspace(0, total_pred_len / frame_rate, total_pred_len)

# ---------- 保存图像 ----------
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True)

for i, label in enumerate(labels):
    for g in range(3):
        index = i + g * 6
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, y_true_full[:, index], label='True', color='blue')
        plt.plot(time_axis, y_pred_full[:, index], label='Predicted', color='red', linestyle='--')
        plt.title(f'{label.upper()}-{groups[g]} Prediction')
        plt.xlabel('Time (s)')
        plt.ylabel(f'{label}_{groups[g]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(save_dir, f'{label}_{groups[g]}_testset.png')
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"✅ 图像已保存：{path}")

# ---------- 误差评估 ----------
print("\n📊 整体预测误差（2000s测试集）：")
for g in range(3):
    print(f"\n【{groups[g].upper()}】")
    for i, label in enumerate(labels):
        index = i + g * 6
        true = y_true_full[:, index]
        pred = y_pred_full[:, index]
        rmse = math.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
        r2 = r2_score(true, pred)
        print(f"{label.upper()} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

# ---------- 保存 CSV ----------
results = []
for t in range(total_pred_len):
    row = {'Time (s)': round(time_axis[t], 2)}
    for g in range(3):
        for i, label in enumerate(labels):
            index = i + g * 6
            row[f'{label}_{groups[g]}_True'] = y_true_full[t, index]
            row[f'{label}_{groups[g]}_Pred'] = y_pred_full[t, index]
    results.append(row)

df_result = pd.DataFrame(results)
df_result.to_csv('cnn_bilstm_attention_multitask_prediction_testset.csv', index=False)
print("✅ 多任务预测结果已保存为 cnn_bilstm_attention_multitask_prediction_testset.csv")

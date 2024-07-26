import numpy as np
import matplotlib.pyplot as plt

# 生成一些模拟数据
np.random.seed(0)
x = np.linspace(0, 10, 50)  # 时间或独立变量
true_values = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)  # 真实值
predicted_values = np.sin(x) + np.random.normal(scale=0.2, size=x.shape)  # 预测值

plt.figure(figsize=(12, 8))

# 绘制真实值
plt.plot(x, true_values, marker='o', linestyle='-', color='blue', label='True Values')

# 绘制预测值
plt.plot(x, predicted_values, marker='x', linestyle='--', color='green', label='Predicted Values')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Comparison of True Values and Predicted Values')
plt.xlabel('X')
plt.ylabel('Values')
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制真实值和预测值
ax1.plot(x, true_values, marker='o', linestyle='-', color='blue', label='True Values')
ax1.plot(x, predicted_values, marker='x', linestyle='--', color='green', label='Predicted Values')
ax1.set_xlabel('X')
ax1.set_ylabel('Values')
ax1.legend(loc='upper left')

# 创建共享x轴的第二个y轴
ax2 = ax1.twinx()

# 计算并绘制误差
errors = predicted_values - true_values
ax2.plot(x, errors, marker='o', linestyle='-', color='red', label='Prediction Error')
ax2.set_ylabel('Error (Predicted - True)')
ax2.legend(loc='upper right')

# 设置图表标题和显示网格
ax1.set_title('True vs Predicted Values and Errors')
# ax1.grid(True)


# 显示网格
plt.grid(True)

plt.show()

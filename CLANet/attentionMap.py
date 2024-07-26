import matplotlib.pyplot as plt
import numpy as np
import torch


def plotTemporalAttentionMap(attention_weight):
    # Attention weight processing
    data_processed = attention_weight.squeeze(axis=-1)
    data = data_processed[0, :]
    data *= 100
    data = data.detach().cpu().numpy()
    # 计算横轴的值，从t-15到t
    x_axis = list(range(-15, 0))  # 即从-15到0，共16个值

    # 创建柱状图
    plt.bar(x_axis, data)

    # 设置图表标题和轴标签
    plt.title('Temporal Attention Weight')
    plt.xlabel('Time Step')
    plt.ylabel('百分比 (%)')
    plt.show()

def plotSpatialAttentionMap(data, mask):
    # Attention weight processing
    data_processed = data.squeeze(axis=-1)
    y_data = data_processed[1, :].tolist()
    y_data_trimmed = y_data[:-1]
    heatmap_data = torch.tensor(y_data_trimmed).reshape(3, 13)
    heatmap_data = heatmap_data.cuda()

    # processing mask matrix
    mask = mask[0, :, :, 0].cuda()
    soc_enc = torch.zeros_like(mask).float()  # mask size: (128, 3, 13, 64)
    soc_enc = soc_enc.masked_scatter_(mask, heatmap_data)

    soc_enc = soc_enc.detach().cpu().numpy()

    plt.figure(figsize=(10, 5))  # 图像大小为 6:3
    plt.imshow(soc_enc, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.xticks(np.arange(13), fontweight='bold')
    plt.yticks(np.arange(3), fontweight='bold')
    for i in range(3):
        for j in range(13):
            plt.text(j, i, f'{soc_enc[i, j] * 100:.2f}%', ha='center', va='center', color='white', fontweight='bold',
                     fontsize=10)
    plt.colorbar().remove()

    plt.title('Heatmap of 3x13 Array', fontweight='bold')
    plt.xlabel('Columns', fontweight='bold')
    plt.ylabel('Rows', fontweight='bold')
    plt.tight_layout()
    # plt.savefig('/path/to/save/heatmap.png', dpi=600)
    plt.show()



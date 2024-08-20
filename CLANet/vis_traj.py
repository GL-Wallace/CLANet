import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
import os


def calculate_rotation_angle(x1, y1, x2, y2):
    """计算从点(x1, y1)到点(x2, y2)的旋转角度"""
    angle = np.arctan2(x2 - x1, y2 - y1)  # 注意这里互换了X和Y
    return np.degrees(angle)


def plot_trajectories(past_traj, fut_traj, nbrs_traj, traj_mask, scale=5, save_imgs=True):
    B, _, _ = past_traj.shape
    N, _, _ = nbrs_traj.shape
    _, G_X, G_Y, _ = traj_mask.shape
    draw_line_type = False

    ego_img = mpimg.imread('1.png')
    nbr_img = mpimg.imread('2.png')

    # 创建保存图像的目录
    if save_imgs and not os.path.exists('result'):
        os.makedirs('result')

    count = 0
    for b in range(B):
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(25, 7))
        # fig, ax = plt.subplots()
        # 绘制自车的历史轨迹
        past_x, past_y = past_traj[b, :, 0], past_traj[b, :, 1]
        ax.plot(past_y, past_x, 'b-o', label='Past Trajectory')  # X和Y交换位置
        # 计算自车图标的旋转角度
        if len(past_x) > 1:
            angle = calculate_rotation_angle(past_x[-2], past_y[-2], past_x[-1], past_y[-1])
            transform = Affine2D().rotate_deg_around(past_y[-1], past_x[-1], angle)
            ax.imshow(ego_img, extent=[past_y[-1] - scale, past_y[-1] + scale, past_x[-1] - scale, past_x[-1] + scale],
                      transform=transform + ax.transData, zorder=2)

        # 绘制自车的预测轨迹
        fut_x, fut_y = fut_traj[b, :, 0], fut_traj[b, :, 1]
        ax.plot(fut_y, fut_x, 'r--', label='Future Trajectory', zorder=1)  # X和Y交换位置

        # 绘制邻车的历史轨迹
        for g1 in range(G_X):
            for g2 in range(G_Y):
                if traj_mask[b, g1, g2, -1].all():
                    if count < N:
                        nbr_x, nbr_y = nbrs_traj[count, :, 0], nbrs_traj[count, :, 1]
                        if draw_line_type is False:
                            ax.plot(nbr_y, nbr_x, 'g-.', label='Neighbor Trajectory')  # X和Y交换位置
                            draw_line_type = True
                        else:
                            ax.plot(nbr_y, nbr_x, 'g-.', zorder=1)  # X和Y交换位置
                        # 计算邻车图标的旋转角度
                        if len(nbr_x) > 1:
                            angle = calculate_rotation_angle(nbr_x[-2], nbr_y[-2], nbr_x[-1], nbr_y[-1])
                            transform = Affine2D().rotate_deg_around(nbr_y[-1], nbr_x[-1], angle)
                            ax.imshow(nbr_img,
                                      extent=[nbr_y[-1] - scale, nbr_y[-1] + scale, nbr_x[-1] - scale,
                                              nbr_x[-1] + scale],
                                      transform=transform + ax.transData, zorder=2)
                        count += 1

        lane_positions = [2.8, -2.8]
        # for lane_y in lane_positions:
        #     ax.axhline(y=lane_y, color='black', linestyle='--', linewidth=1)
        # 设置坐标轴范围以自适应轨迹范围
        all_x = np.concatenate([past_x, fut_x, nbrs_traj[:N, :, 0].flatten()])
        all_y = np.concatenate([past_y, fut_y, nbrs_traj[:N, :, 1].flatten()])
        ax.set_xlim(all_y.min(), all_y.max())
        ax.set_ylim(all_x.min(), all_x.max())

        ax.set_title(f'Scene {b + 1}')
        ax.set_xlabel('Y Coordinate')  # X坐标变为Y轴
        ax.set_ylabel('X Coordinate')  # Y坐标变为X轴

        ax.set_xticks(np.arange(-250, 300, 50))
        ax.set_yticks(np.arange(-20, 25, 10))

        ax.legend()
        ax.grid(False)

        if save_imgs:
            plt.savefig(f'result/scene_{b + 1}.png')
            plt.close(fig)
        else:
            plt.show()


# 示例数据
past_traj = np.random.rand(2, 16, 2)  # B=2, 16点
fut_traj = np.random.rand(2, 24, 2)  # B=2, 24点

plot_trajectories(past_traj, fut_traj, np.random.rand(2, 16, 2), np.ones((2, 2, 2, 1)), scale=5, save_imgs=True)

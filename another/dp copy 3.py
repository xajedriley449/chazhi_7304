# 第四部分：生成三维空间微震事件分布及其三视图投影（原始坐标 + 对数能量）
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

output_dir = 'another/data_preprocessing copy 2'
os.makedirs(output_dir, exist_ok=True)

# 毕业论文统一图表设置（字体、字号）
# 毕业论文统一图表设置（字体、字号）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'SimSun'],
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    # 【致胜修改】：全面提高所有元素的基准字号
    'font.size': 16,               # 基础字号从 12 提升到 14
    'axes.titlesize': 20,          # 标题字号 18
    'axes.labelsize': 18,          # 坐标轴标签字号 16
    'xtick.labelsize': 16,         # 刻度数字 14
    'ytick.labelsize': 16,
    'svg.fonttype': 'path',        # 文字转轮廓，保证中文显示
})

# 读取数据
df = pd.read_csv('data/7303_xyzet_rotated.csv')  # 替换为实际文件路径


# ------------------ 数据准备 ------------------
# 假设已有 DataFrame df 包含字段：x, y, z, e
x_orig = df['x'].values / 1000.0      # 走向坐标，单位 km
y_orig = df['y'].values / 1000.0      # 倾向坐标，单位 km
z_orig = df['z'].values / 1000.0      # 垂向坐标，单位 km
e_orig = df['e'].values

# 按原始能量从小到大排序（小能量先绘制，大能量后绘制覆盖在上面）
sort_idx = np.argsort(e_orig)
x_orig = x_orig[sort_idx]
y_orig = y_orig[sort_idx]
z_orig = z_orig[sort_idx]
e_orig = e_orig[sort_idx]

# 计算对数能量 E_i = lg(e_i + 1)
E_log = np.log10(e_orig + 1.0)

cmap = plt.get_cmap('turbo')  # 选择颜色映射方案（可选 'viridis', 'plasma' 等）

# 点大小缩放因子（可根据效果调整）
size_scale = 1 #50
# E_log = (E_log - E_log.min()) / (E_log.max() - E_log.min())  # 归一化到 [0, 1]
point_sizes = size_scale * E_log   # 加常数避免最小点过小
size_scale = 0.01 #50
# e_orig = (e_orig - e_orig.min()) / (e_orig.max() - e_orig.min())  # 归一化到 [0, 1]
point_sizes_orig = size_scale * e_orig  # 原始能量点大小（可选）

# 计算各维度的实际物理范围（用于设置等比例）
x_range = x_orig.max() - x_orig.min()
y_range = y_orig.max() - y_orig.min()
z_range = z_orig.max() - z_orig.min()

# 计算一个统一的边距比例（让图形不要顶到边框）
margin = 0.05
x_min, x_max = x_orig.min() - margin * x_range, x_orig.max() + margin * x_range
y_min, y_max = y_orig.min() - margin * y_range, y_orig.max() + margin * y_range
z_min, z_max = z_orig.min() - margin * z_range, z_orig.max() + margin * z_range

# ------------------ 绘图 ------------------
# ------------------ 透明度设置 ------------------
ALPHA = 0.8  

# =============================================================================
# 统一绘图与输出辅助函数（针对性分离 3D 与 2D 的排版逻辑）
# =============================================================================
def plot_and_save_single_view(
    save_name, view_type, x_data, y_data, z_data, s_data, c_data, cmap,
    x_lim, y_lim, z_lim, x_label, y_label, z_label, title, cbar_label,
    is_3d=False, cbar_ticks=None, cbar_ticklabels=None, show_title=False
):
    # 【修改】：缩小画布的物理尺寸！当插入 Word 时，较小的原图不需要怎么缩放，字体就显得非！常！大！
    fig = plt.figure(figsize=(7.5, 5.5))  
    
    if is_3d:
        # ======= 3D 图形排版逻辑 =======
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_data, y_data, z_data, s=s_data, c=c_data, cmap=cmap, alpha=ALPHA)
        
        # 调整 3D 下的大字号留白防重叠
        ax.set_zlabel(z_label, labelpad=3)
        ax.set_xlabel(x_label, labelpad=22)
        ax.set_ylabel(y_label, labelpad=8)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        
        ax.set_box_aspect([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]])
        ax.zaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(useOffset=False, style='plain', axis='both')
        # 刻度标签稍微设小一点点防拥挤
        ax.tick_params(axis='x', labelsize=12) 
        ax.tick_params(axis='y', labelsize=12) 
        ax.tick_params(axis='z', labelsize=12) 
        if show_title:
            ax.set_title(title, y=0.88)
        
        if cbar_ticks is not None:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.15, fraction=0.03, shrink=0.7, ticks=cbar_ticks)
            cbar.ax.set_yticklabels(cbar_ticklabels, rotation=40, ha='left', fontsize=12) 
        else:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.15, fraction=0.03, shrink=0.7)
            cbar.ax.tick_params(labelsize=14)
        cbar.set_label(cbar_label, labelpad=10, fontsize=16)

    else:
        # ======= 2D 图形排版逻辑 =======
        ax = fig.add_subplot(111)
        scatter = ax.scatter(x_data, y_data, s=s_data, c=c_data, cmap=cmap, alpha=ALPHA)
        
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal') 
        ax.ticklabel_format(useOffset=False, style='plain')
        
        # 【新增修改】：将 2D 图的 X 轴刻度标签倾斜排布，避免缩放后数字互相挤压重叠
        ax.tick_params(axis='x', rotation=40)
        # 旋转后将对齐锚点设在右侧，使其完美指向刻度线
        for label in ax.get_xticklabels():
            label.set_ha('right')
        
        if show_title:
            ax.set_title(title, pad=15)
        
        if cbar_ticks is not None:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.05, fraction=0.046, shrink=0.8, ticks=cbar_ticks)
            cbar.ax.set_yticklabels(cbar_ticklabels, rotation=40, ha='left', fontsize=12)
        else:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.05, fraction=0.046, shrink=0.8)
            cbar.ax.tick_params(labelsize=14)
        cbar.set_label(cbar_label, labelpad=12, fontsize=16)
    
    plt.tight_layout()
    
    # 导出
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=600)
    png_path = save_path.replace('.svg', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"[{view_type}] 已保存至: {save_path}")
    plt.close(fig)


# =============================================================================
# 版本一：对数能量版 (颜色和大小均由对数能量控制)
# =============================================================================
print("\n--- 正在生成 版本一：对数能量分布图 ---")
v1_configs = [
    ('v1_logE_3D.svg', '3D主视图', x_orig, y_orig, z_orig, x_min, x_max, y_min, y_max, True, 'X (km)', 'Y (km)', '微震事件三维分布 (对数能量)'),
    ('v1_logE_XY.svg', 'XY俯视图', x_orig, y_orig, None, x_min, x_max, y_min, y_max, False, 'X (km)', 'Y (km)', '对数能量 X-Y 平面投影 (俯视图)'),
    ('v1_logE_XZ.svg', 'XZ走向剖面', x_orig, z_orig, None, x_min, x_max, z_min, z_max, False, 'X (km)', 'Z (km)', '对数能量 X-Z 平面投影 (走向剖面)'),
    ('v1_logE_YZ.svg', 'YZ倾向剖面', y_orig, z_orig, None, y_min, y_max, z_min, z_max, False, 'Y (km)', 'Z (km)', '对数能量 Y-Z 平面投影 (倾向剖面)')
]

for fname, vtype, d_x, d_y, d_z, lim_x, lim_X, lim_y, lim_Y, is3d, lab_x, lab_y, tit in v1_configs:
    plot_and_save_single_view(
        fname, vtype, d_x, d_y, d_z, point_sizes, E_log, cmap,
        (lim_x, lim_X), (lim_y, lim_Y), (z_min, z_max),
        lab_x, lab_y, 'Z (km)', tit, '对数能量 lg(e+1) (J)', is_3d=is3d
    )


# =============================================================================
# 版本二：原始能量版 (颜色控制和大小受原始能量影响)
# =============================================================================
print("\n--- 正在生成 版本二：原始能量分布图 ---")
x_orig_v2, y_orig_v2, z_orig_v2 = df['x'].values/1000.0, df['y'].values/1000.0, df['z'].values/1000.0
sort_idx_v2 = np.argsort(df['e'].values)
x_srt, y_srt, z_srt, e_srt = x_orig_v2[sort_idx_v2], y_orig_v2[sort_idx_v2], z_orig_v2[sort_idx_v2], df['e'].values[sort_idx_v2]

v2_configs = [
    ('v2_rawE_3D.svg', '3D主视图', x_srt, y_srt, z_srt, x_min, x_max, y_min, y_max, True, 'X (km)', 'Y (km)', '微震事件三维分布 (原始能量)'),
    ('v2_rawE_XY.svg', 'XY俯视图', x_srt, y_srt, None, x_min, x_max, y_min, y_max, False, 'X (km)', 'Y (km)', '原始能量 X-Y 面投影 (俯视图)'),
    ('v2_rawE_XZ.svg', 'XZ走向剖面', x_srt, z_srt, None, x_min, x_max, z_min, z_max, False, 'X (km)', 'Z (km)', '原始能量 X-Z 面投影 (走向剖面)'),
    ('v2_rawE_YZ.svg', 'YZ倾向剖面', y_srt, z_srt, None, y_min, y_max, z_min, z_max, False, 'Y (km)', 'Z (km)', '原始能量 Y-Z 面投影 (倾向剖面)')
]

for fname, vtype, d_x, d_y, d_z, lim_x, lim_X, lim_y, lim_Y, is3d, lab_x, lab_y, tit in v2_configs:
    plot_and_save_single_view(
        fname, vtype, d_x, d_y, d_z, point_sizes_orig, e_srt, cmap,
        (lim_x, lim_X), (lim_y, lim_Y), (z_min, z_max),
        lab_x, lab_y, 'Z (km)', tit, '原始能量 e (J)', is_3d=is3d
    )


# =============================================================================
# 版本三：时间颜色版 (大小为对数能量，颜色为发生时间)
# =============================================================================
print("\n--- 正在生成 版本三：时间演化动力学分布图 ---")
from datetime import datetime

df_full = pd.read_csv('data/7303_all.csv', encoding='gb2312')
df_full['datetime'] = pd.to_datetime(df_full['日期'] + ' ' + df_full['时间'], format='%Y/%m/%d %H:%M:%S')

x_v3, y_v3, z_v3 = df['x'].values/1000.0, df['y'].values/1000.0, df['z'].values/1000.0
e_v3, time_v3 = df_full['能量[J]'].values, df_full['datetime'].values

sort_idx_v3 = np.argsort(e_v3)
x_s3, y_s3, z_s3 = x_v3[sort_idx_v3], y_v3[sort_idx_v3], z_v3[sort_idx_v3]
e_s3, time_s3 = e_v3[sort_idx_v3], time_v3[sort_idx_v3]

point_sizes_v3 = 1.0 * np.log10(e_s3 + 1.0)
time_numeric = time_s3.astype('int64') // 10**9
time_norm = (time_numeric - time_numeric.min()) / (time_numeric.max() - time_numeric.min())

cmap_time = plt.get_cmap('viridis')

# 构造日期刻度
num_ticks = 7
tick_timestamps = np.linspace(time_numeric.min(), time_numeric.max(), num_ticks)

# 【修改点】：将 '%Y/%m/%d' 改为无前导零的 '%Y/%-m/%-d' 格式 (Windows 下为 '%Y/%#m/%#d')
import os
fmt_str = '%Y/%#m/%#d' if os.name == 'nt' else '%Y/%-m/%-d'
tick_dates = [datetime.fromtimestamp(ts).strftime(fmt_str) for ts in tick_timestamps]

tick_positions = (tick_timestamps - time_numeric.min()) / (time_numeric.max() - time_numeric.min())

v3_configs = [
    ('v3_time_3D.svg', '3D主视图', x_s3, y_s3, z_s3, x_min, x_max, y_min, y_max, True, 'X (km)', 'Y (km)', '微震事件三维分布 (时空演化)'),
    ('v3_time_XY.svg', 'XY俯视图', x_s3, y_s3, None, x_min, x_max, y_min, y_max, False, 'X (km)', 'Y (km)', '时空演化 X-Y 面投影 (俯视图)'),
    ('v3_time_XZ.svg', 'XZ走向剖面', x_s3, z_s3, None, x_min, x_max, z_min, z_max, False, 'X (km)', 'Z (km)', '时空演化 X-Z 面投影 (走向剖面)'),
    ('v3_time_YZ.svg', 'YZ倾向剖面', y_s3, z_s3, None, y_min, y_max, z_min, z_max, False, 'Y (km)', 'Z (km)', '时空演化 Y-Z 面投影 (倾向剖面)')
]

for fname, vtype, d_x, d_y, d_z, lim_x, lim_X, lim_y, lim_Y, is3d, lab_x, lab_y, tit in v3_configs:
    plot_and_save_single_view(
        fname, vtype, d_x, d_y, d_z, point_sizes_v3, time_norm, cmap_time,
        (lim_x, lim_X), (lim_y, lim_Y), (z_min, z_max),
        lab_x, lab_y, 'Z (km)', tit, '微震发生时间 (年/月/日)', 
        is_3d=is3d, cbar_ticks=tick_positions, cbar_ticklabels=tick_dates
    )

print("\n🎉 所有图像处理并导出完毕！")
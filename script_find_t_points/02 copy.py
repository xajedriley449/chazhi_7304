import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# 全局绘图参数（严格对齐论文大字号排版）
plt.rcParams.update({
    'font.size': 12,               
    'axes.labelsize': 18,          
    'xtick.labelsize': 16,         
    'ytick.labelsize': 16,         
})

def apply_dbscan_clustering(df, eps=0.01, min_samples=5):
    # 初始化所有点为噪声点(-1)
    df['cluster'] = -1
    
    # 只对第一阶段识别为连线点的数据进行DBSCAN
    points_of_interest = df[df['is_dense_point'] == 1]
    
    if len(points_of_interest) == 0:
        return df
        
    # 准备数据
    X = points_of_interest[['x', 'y', 'z', 't']].values
    
    # 归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # 应用DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_normalized)
    
    # 只更新第一阶段连线点的聚类结果
    df.loc[points_of_interest.index, 'cluster'] = clusters
    
    return df

def visualize_matplotlib_style(df, result_dir):
    """
    生成归一化多视图投影（不同聚类簇采用不同颜色，噪声点为灰）
    """
    img_dir = os.path.join(result_dir, '02_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 坐标系空间及时间归一化可视化
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(df[['x']]).flatten()
    y_norm = scaler.fit_transform(df[['y']]).flatten()
    z_norm = scaler.fit_transform(df[['z']]).flatten()
    t_norm = scaler.fit_transform(df[['t']]).flatten()  # 【新增】：提取并归一化时间 T
    
    # 【新增修改】：采用原 02 脚本的多彩聚类簇配色逻辑
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    color_dict = {-1: 'gray'}  # 噪声点为灰色
    for i, cluster in enumerate(unique_clusters):
        color_dict[cluster] = colors[i]
        
    point_colors = df['cluster'].map(color_dict).values
    
    print(f"\n生成多彩聚类网络可视化投影 (彩色=成功聚类的时空簇，灰=被过滤的噪声点)...")
    
    # ================= 1. 三维视图 (X-Y-Z) =================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_norm, y_norm, z_norm, c=point_colors, s=2, alpha=0.6)
    
    # 设置防拥挤边距
    ax.set_xlabel('X', labelpad=25)
    ax.set_ylabel('Y', labelpad=25)
    ax.set_zlabel('Z', labelpad=20)
    
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=8)
    
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-45)  # 保证Z轴在右侧
    ax.tick_params(axis='both', labelsize=12)
    
    save_path = os.path.join(img_dir, 'XYZ_3D_projection.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"已保存: {save_path}")

    # ================= 2. 二维视图 (XY, XZ, YZ, XT, YT, ZT) =================
    # 【新增】：扩展投影列表，加入时间(T)为横轴，空间(X/Y/Z)为纵轴的时空视图
    projections = [
        ('XY', x_norm, y_norm, 'X', 'Y'),
        ('XZ', x_norm, z_norm, 'X', 'Z'),
        ('YZ', y_norm, z_norm, 'Y', 'Z'),
        ('XT', t_norm, x_norm, 'T', 'X'),
        ('YT', t_norm, y_norm, 'T', 'Y'),
        ('ZT', t_norm, z_norm, 'T', 'Z')
    ]
    
    for title, px, py, xlabel, ylabel in projections:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(px, py, c=point_colors, s=2, alpha=0.6)
        
        ax.set_xlabel(f'{xlabel}', labelpad=10)
        ax.set_ylabel(f'{ylabel}', labelpad=10)
        ax.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        save_path = os.path.join(img_dir, f'{title}_projection.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path}")

def main():
    # 读取第一阶段结果
    input_path = 'script_find_t_points/result/density_classification_results.csv'
    output_dir = 'script_find_t_points/result'
    
    if not os.path.exists(input_path):
        print(f"找不到第一阶段结果，请先运行 01 copy.py")
        return
        
    print("加载第一阶段高密度点分类数据...")
    df = pd.read_csv(input_path)
    
    print("执行时空 DBSCAN 聚类过滤...")
    # 应用DBSCAN聚类
    df_clustered = apply_dbscan_clustering(df, eps=0.01, min_samples=5)
    
    # 保存结果
    output_path = os.path.join(output_dir, 'dbscan_clustering_results.csv')
    df_clustered.to_csv(output_path, index=False)
    print(f"数据已保存至: {output_path}")
    
    # 绘制结果
    visualize_matplotlib_style(df_clustered, output_dir)
    print("\n🎉 第二阶段聚类清洗及可视化完成！图表位于 script_find_t_points/result/02_img/ 下")

if __name__ == "__main__":
    main()
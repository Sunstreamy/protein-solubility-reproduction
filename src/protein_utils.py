import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def create_adjacency_matrix_from_3d_structure(
    atomic_coordinates, distance_threshold=10.0
):
    """
    从蛋白质3D结构坐标创建GCN的邻接矩阵

    参数:
    -----------
    atomic_coordinates : numpy.ndarray
        形状为(n_residues, 3)的数组，包含每个残基的Cα原子的3D坐标
    distance_threshold : float
        考虑两个残基相连的阈值距离(单位：埃)

    返回:
    --------
    adjacency_matrix : numpy.ndarray
        形状为(n_residues, n_residues)的二进制邻接矩阵
    distance_map : numpy.ndarray
        形状为(n_residues, n_residues)的距离图
    """
    # 计算所有残基之间的欧氏距离
    distance_map = squareform(pdist(atomic_coordinates, "euclidean"))

    # 根据距离阈值创建二进制邻接矩阵
    adjacency_matrix = np.zeros_like(distance_map)
    adjacency_matrix[distance_map <= distance_threshold] = 1

    # 移除自连接(对角线元素)
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix, distance_map


def visualize_distance_map(distance_map, title="残基距离图"):
    """
    将距离图可视化为2D热图
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_map, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="距离(Å)")
    plt.title(title)
    plt.xlabel("残基索引")
    plt.ylabel("残基索引")
    plt.tight_layout()
    plt.show()


def normalize_adjacency_matrix(adjacency_matrix):
    """
    按照论文中的描述归一化邻接矩阵用于GCN:
    A = D^(-1/2) * A * D^(-1/2)，其中D是度矩阵
    """
    # 为邻接矩阵添加自环
    adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])

    # 计算度矩阵
    degree_matrix = np.sum(adjacency_matrix, axis=1)

    # 计算D^(-1/2)
    degree_matrix_inv_sqrt = np.diag(np.power(degree_matrix, -0.5))

    # 计算归一化邻接矩阵
    normalized_adjacency = np.matmul(
        np.matmul(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt
    )

    return normalized_adjacency

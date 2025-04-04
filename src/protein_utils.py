import numpy as np
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser
import os
import glob


def validate_matrices(distance_map, adjacency_matrix, normalized_adjacency):
    """验证生成的矩阵是否正确"""
    print("\n矩阵验证结果:")
    print("-" * 50)

    # 1. 检查距离图
    print("\n距离图验证:")
    print(f"形状: {distance_map.shape}")
    print(f"对角线元素是否为零: {np.allclose(np.diag(distance_map), 0)}")
    print(f"矩阵是否对称: {np.allclose(distance_map, distance_map.T)}")
    print(f"距离范围: [{np.min(distance_map):.2f}, {np.max(distance_map):.2f}] Å")

    # 2. 检查邻接矩阵
    print("\n邻接矩阵验证:")
    print(f"形状: {adjacency_matrix.shape}")
    print(f"对角线元素是否为1: {np.allclose(np.diag(adjacency_matrix), 1)}")
    print(f"矩阵是否对称: {np.allclose(adjacency_matrix, adjacency_matrix.T)}")
    print(f"值范围: [{np.min(adjacency_matrix):.2f}, {np.max(adjacency_matrix):.2f}]")

    # 3. 检查归一化邻接矩阵
    print("\n归一化邻接矩阵验证:")
    print(f"形状: {normalized_adjacency.shape}")
    print(f"矩阵是否对称: {np.allclose(normalized_adjacency, normalized_adjacency.T)}")
    print(
        f"值范围: [{np.min(normalized_adjacency):.2f}, {np.max(normalized_adjacency):.2f}]"
    )

    # 4. 检查矩阵之间的关系
    print("\n矩阵关系验证:")
    # 检查距离和邻接值是否呈负相关
    # 将矩阵展平并移除对角线元素
    mask = ~np.eye(distance_map.shape[0], dtype=bool)
    distances_flat = distance_map[mask]
    adjacency_flat = adjacency_matrix[mask]
    correlation = np.corrcoef(distances_flat, adjacency_flat)[0, 1]
    print(f"距离-邻接相关性: {correlation:.3f} (应为负值)")


def extract_ca_coordinates_from_pdb(pdb_file):
    """从PDB文件中提取Cα原子的坐标"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # 提取第一个模型
    model = structure[0]

    ca_coords = []
    residue_ids = []

    # 遍历所有残基
    for chain in model:
        for residue in chain:
            # 检查是否含有Cα原子
            if "CA" in residue:
                ca_atom = residue["CA"]
                ca_coords.append(ca_atom.get_coord())
                residue_ids.append((chain.id, residue.id[1]))

    return np.array(ca_coords), residue_ids


def create_adjacency_matrix(ca_coords, d0=4.0):
    """
    根据论文中的公式，从Cα坐标计算GCN的邻接矩阵
    S_ij = 2/(1 + max(d0, d_ij)/d0)
    """
    # 计算所有残基之间的欧氏距离
    distance_map = squareform(pdist(ca_coords, "euclidean"))

    # 创建邻接矩阵
    n_residues = len(ca_coords)
    adjacency_matrix = np.zeros((n_residues, n_residues))

    # 根据论文中的公式计算邻接矩阵
    for i in range(n_residues):
        for j in range(n_residues):
            dij = distance_map[i, j]
            max_val = max(d0, dij)
            adjacency_matrix[i, j] = 2 / (1 + max_val / d0)

    return adjacency_matrix, distance_map


def normalize_adjacency_matrix(adjacency_matrix):
    """对邻接矩阵进行归一化，用于GCN：A = D^(-1/2) * A * D^(-1/2)"""
    # 计算度矩阵
    degree_matrix = np.sum(adjacency_matrix, axis=1)

    # 计算D^(-1/2)
    degree_matrix_inv_sqrt = np.diag(np.power(degree_matrix, -0.5))

    # 计算归一化邻接矩阵
    normalized_adjacency = np.matmul(
        np.matmul(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt
    )

    return normalized_adjacency


def process_protein_structure(
    pdb_file, adj_dir, norm_adj_dir, d0=4.0, skip_existing=False
):
    """
    处理蛋白质结构文件，生成GCN的邻接矩阵

    参数:
    -----------
    pdb_file : str
        PDB文件路径
    adj_dir : str
        邻接矩阵保存目录
    norm_adj_dir : str
        归一化邻接矩阵保存目录
    d0 : float
        距离阈值，默认为4.0埃
    skip_existing : bool
        是否跳过已存在的文件，默认为False

    返回:
    --------
    normalized_adjacency : numpy.ndarray
        归一化后的邻接矩阵，用于GCN
    """
    # 创建输出目录
    os.makedirs(adj_dir, exist_ok=True)
    os.makedirs(norm_adj_dir, exist_ok=True)

    # 提取蛋白质名称
    protein_name = os.path.basename(pdb_file).split(".")[0]

    # 检查文件是否已存在
    adj_file = f"{adj_dir}/{protein_name}_adjacency.npy"
    norm_adj_file = f"{norm_adj_dir}/{protein_name}_normalized_adjacency.npy"

    if skip_existing and os.path.exists(adj_file) and os.path.exists(norm_adj_file):
        print(f"跳过 {protein_name}，文件已存在")
        # 加载已存在的文件
        adjacency_matrix = np.load(adj_file)
        normalized_adjacency = np.load(norm_adj_file)
        # 重新计算距离图用于验证
        ca_coords, residue_ids = extract_ca_coordinates_from_pdb(pdb_file)
        _, distance_map = create_adjacency_matrix(ca_coords, d0)
        return normalized_adjacency, distance_map, ca_coords, residue_ids

    # 1. 提取Cα原子坐标
    ca_coords, residue_ids = extract_ca_coordinates_from_pdb(pdb_file)
    print(f"成功提取 {len(ca_coords)} 个残基的Cα坐标")

    # 2. 创建邻接矩阵和距离图
    adjacency_matrix, distance_map = create_adjacency_matrix(ca_coords, d0)
    print(f"成功创建邻接矩阵，形状: {adjacency_matrix.shape}")

    # 3. 归一化邻接矩阵
    normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
    print(f"成功归一化邻接矩阵")

    # 保存结果为NumPy数组 - 分别保存到不同目录
    np.save(adj_file, adjacency_matrix)
    np.save(norm_adj_file, normalized_adjacency)

    print(f"已保存邻接矩阵到 {adj_file}")
    print(f"已保存归一化邻接矩阵到 {norm_adj_file}")

    # 验证矩阵
    validate_matrices(distance_map, adjacency_matrix, normalized_adjacency)

    return normalized_adjacency, distance_map, ca_coords, residue_ids


# 处理目录中的所有PDB文件
if __name__ == "__main__":
    # PDB文件目录
    pdb_dir = "/home/sunstreamy/code/project/AI/protein-solubility-reproduction/paperAbout/dataset/pdb"

    # 邻接矩阵和归一化邻接矩阵的保存目录
    adj_dir = "/home/sunstreamy/code/project/AI/protein-solubility-reproduction/adjacency_matrices"
    norm_adj_dir = "/home/sunstreamy/code/project/AI/protein-solubility-reproduction/normalized_adjacency_matrices"

    # 跳过已存在的文件
    skip_existing = True

    # 获取目录中所有PDB文件
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))

    print(f"找到 {len(pdb_files)} 个PDB文件")

    # 处理每个PDB文件
    for i, pdb_file in enumerate(pdb_files):
        protein_name = os.path.basename(pdb_file).split(".")[0]
        print(f"\n处理第 {i+1}/{len(pdb_files)} 个蛋白质: {protein_name}")

        try:
            # 处理蛋白质结构
            normalized_adjacency, distance_map, ca_coords, residue_ids = (
                process_protein_structure(
                    pdb_file, adj_dir, norm_adj_dir, skip_existing=skip_existing
                )
            )

            print(f"处理完成: {os.path.basename(pdb_file)}")
            print(f"蛋白质包含 {len(ca_coords)} 个残基")
            print(f"邻接矩阵形状: {normalized_adjacency.shape}")
        except Exception as e:
            print(f"处理 {protein_name} 时出错: {e}")

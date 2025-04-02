import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import os

def extract_ca_coordinates_from_pdb(pdb_file):
    """从PDB文件中提取Cα原子的坐标"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 提取第一个模型
    model = structure[0]
    
    ca_coords = []
    residue_ids = []
    
    # 遍历所有残基
    for chain in model:
        for residue in chain:
            # 检查是否含有Cα原子
            if 'CA' in residue:
                ca_atom = residue['CA']
                ca_coords.append(ca_atom.get_coord())
                residue_ids.append((chain.id, residue.id[1]))
    
    return np.array(ca_coords), residue_ids

def create_adjacency_matrix(ca_coords, d0=4.0):
    """
    根据论文中的公式，从Cα坐标计算GCN的邻接矩阵
    S_ij = 2/(1 + max(d0, d_ij)/d0)
    """
    # 计算所有残基之间的欧氏距离
    distance_map = squareform(pdist(ca_coords, 'euclidean'))
    
    # 创建邻接矩阵
    n_residues = len(ca_coords)
    adjacency_matrix = np.zeros((n_residues, n_residues))
    
    # 根据论文中的公式计算邻接矩阵
    for i in range(n_residues):
        for j in range(n_residues):
            dij = distance_map[i, j]
            max_val = max(d0, dij)
            adjacency_matrix[i, j] = 2 / (1 + max_val/d0)
    
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

def visualize_matrix(matrix, title, output_file=None, colorbar_label=""):
    """将矩阵可视化为热图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel('残基索引')
    plt.ylabel('残基索引')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()
    plt.close()

def process_protein_structure(pdb_file, output_dir="./output", d0=4.0):
    """
    完整处理蛋白质结构文件，生成GCN的邻接矩阵
    
    参数:
    -----------
    pdb_file : str
        PDB文件路径
    output_dir : str
        结果保存目录
    d0 : float
        距离阈值，默认为4.0埃
    
    返回:
    --------
    normalized_adjacency : numpy.ndarray
        归一化后的邻接矩阵，用于GCN
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取蛋白质名称
    protein_name = os.path.basename(pdb_file).split('.')[0]
    
    # 1. 提取Cα原子坐标
    ca_coords, residue_ids = extract_ca_coordinates_from_pdb(pdb_file)
    print(f"成功提取 {len(ca_coords)} 个残基的Cα坐标")
    
    # 2. 创建邻接矩阵和距离图
    adjacency_matrix, distance_map = create_adjacency_matrix(ca_coords, d0)
    print(f"成功创建邻接矩阵，形状: {adjacency_matrix.shape}")
    
    # 3. 归一化邻接矩阵
    normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
    print(f"成功归一化邻接矩阵")
    
    # 4. 可视化并保存结果
    # 可视化距离图
    visualize_matrix(
        distance_map, 
        title=f"蛋白质 {protein_name} 残基距离图",
        output_file=f"{output_dir}/{protein_name}_distance_map.png",
        colorbar_label="距离(Å)"
    )
    
    # 可视化邻接矩阵
    visualize_matrix(
        adjacency_matrix, 
        title=f"蛋白质 {protein_name} 邻接矩阵",
        output_file=f"{output_dir}/{protein_name}_adjacency.png",
        colorbar_label="连接强度"
    )
    
    # 可视化归一化邻接矩阵
    visualize_matrix(
        normalized_adjacency, 
        title=f"蛋白质 {protein_name} 归一化邻接矩阵",
        output_file=f"{output_dir}/{protein_name}_normalized_adjacency.png",
        colorbar_label="归一化连接强度"
    )
    
    # 保存结果为NumPy数组
    np.save(f"{output_dir}/{protein_name}_distance_map.npy", distance_map)
    np.save(f"{output_dir}/{protein_name}_adjacency.npy", adjacency_matrix)
    np.save(f"{output_dir}/{protein_name}_normalized_adjacency.npy", normalized_adjacency)
    
    print(f"已保存所有结果到 {output_dir} 目录")
    
    return normalized_adjacency, distance_map, ca_coords, residue_ids

# 使用示例
if __name__ == "__main__":
    # 替换为您的PDB文件路径
    pdb_file = "example_protein.pdb"
    
    # 处理蛋白质结构
    normalized_adjacency, distance_map, ca_coords, residue_ids = process_protein_structure(pdb_file)
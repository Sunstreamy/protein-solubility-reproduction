# 蛋白质溶解度预测模型复现

该项目旨在复现论文"Predicting the effects of mutations on protein solubility using graph convolution network and protein language model representation"中的关键组件。

## 待实现的三个关键任务

1. **蛋白质三维结构的 GCN 邻接矩阵表示**：使用预测的蛋白质三维结构，通过二维距离图来表示 GCN 邻接矩阵
2. **使用 ProtTrans 进行序列特征提取**：通过 ProtTrans 模型对蛋白质序列进行特征提取
3. **LSTM 溶解度预测模型**：训练一个蛋白质溶解度预测模型 LSTM 来提取溶解度相关特征

## 相关仓库

- [GraphSol](https://github.com/jcchan23/GraphSol)：使用 GCN 和接触图预测蛋白质溶解度
- [DeepMutSol](https://github.com/biomed-AI/DeepMutSol)：预测突变对蛋白质溶解度的影响
- [AlphaFold2](https://github.com/google-deepmind/alphafold)：用于预测蛋白质 3D 结构

## 论文中提到但原仓库缺失的数据集

论文中提到但在 DeepMutSol 仓库中可能不完整或缺失的关键数据集：

1. **完整的 eSOL 数据集** - 论文提到使用了约 4132 个大肠杆菌蛋白质的溶解度数据，但 DeepMutSol 仓库中可能只包含部分经过处理的数据。完整数据集可从[eSOL 数据库](https://www.tanpaku.org/tp-esol/)获取。

2. **S. cerevisiae 数据集** - 用于独立测试的酵母菌蛋白质溶解度数据。DeepMutSol 仓库中可能未完整提供，需要参考 GraphSol 论文或从[相关研究资源](https://www.jstage.jst.go.jp/article/biochemistry/108/5/108_12/)获取。

3. **部分蛋白质的 PDB 结构文件** - 论文使用了 AlphaFold2 预测的蛋白质结构，但仓库中可能只包含部分蛋白质的结构文件。缺失的结构可从[AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)下载或使用[ColabFold](https://github.com/sokrypton/ColabFold)预测。

## 数据集与任务对应关系

### 任务 1: 蛋白质三维结构的 GCN 邻接矩阵表示

**所需数据集**:

- **PDB 文件** (位于 DeepMutSol 仓库的`/pdb`目录)：含有蛋白质的原子坐标信息
- **缺失数据**：部分蛋白质的 PDB 文件在 DeepMutSol 仓库中可能不完整，可从[AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)下载补充，或使用[ColabFold](https://github.com/sokrypton/ColabFold)在线预测

**数据处理流程**:

- 从 PDB 文件中提取 Cα 原子坐标
- 计算残基之间的欧氏距离构建距离图
- 基于距离阈值生成二进制邻接矩阵
- 归一化邻接矩阵用于 GCN

### 任务 2: 使用 ProtTrans 进行序列特征提取

**所需数据集**:

- **蛋白质序列数据** (位于 DeepMutSol 仓库的 Excel 文件，如`all_protein_sequences.xlsx`)
- **缺失数据**：ProtTrans 预训练模型需从[Hugging Face](https://huggingface.co/Rostlab)下载，主要包括:
  - [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)
  - [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)
  - [Rostlab/prot_electra_discriminator](https://huggingface.co/Rostlab/prot_electra_discriminator)

**数据处理流程**:

- 加载蛋白质序列
- 使用 ProtTrans 模型提取序列的潜在表示
- 应用不同的池化策略将 token 级特征转换为序列级特征

### 任务 3: LSTM 溶解度预测模型

**所需数据集**:

- **溶解度标签数据** (位于 DeepMutSol 仓库的 Excel 文件，如`train_dataset.xlsx`、`test_dataset.xlsx`)
The datasets are freely available in VariBench database [52,53] at http://structure.bmc.lu.se/VariBench/ponsol2.php (accessed on 26 July 2021).
- DeepMutSol 仓库中可能只包含部分数据

**数据处理流程**:

- 整合任务 1 和任务 2 生成的特征
- 划分训练集和测试集
- 训练 LSTM 模型预测蛋白质溶解度
- 在独立测试集上评估模型性能

## 数据获取方式

1. **论文原始仓库中的数据**
   - 从[DeepMutSol](https://github.com/biomed-AI/DeepMutSol)仓库的`dataset`目录下载：
     - PDB 文件：位于`/pdb`目录
     - 序列和溶解度数据：位于多个 xlsx 文件中
2. **补充数据来源**

   - 蛋白质结构：[AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)或使用[ColabFold](https://github.com/sokrypton/ColabFold)在线预测
   - ProtTrans 预训练模型：[Hugging Face Rostlab 模型仓库](https://huggingface.co/Rostlab)

3. **GraphSol 仓库中的参考数据**
   - 从[GraphSol](https://github.com/jcchan23/GraphSol)仓库下载节点特征和边特征：
     - 节点特征：`./Data/node_features.zip`
     - 边特征：`./Data/edge_features.zip`
     - FASTA 文件：`./Data/fasta.zip`

## 项目结构

```
.
├── README.md
├── requirements.txt
└── src/
    ├── protein_utils.py   # 处理蛋白质结构，生成邻接矩阵
    ├── prot_trans.py      # ProtTrans特征提取
    └── solubility_model.py # 溶解度预测模型
```

## 参考文献

- Wang, J., Chen, S., Yuan, Q., Chen, J., Li, D., Wang, L., & Yang, Y. (2024). Predicting the effects of mutations on protein solubility using graph convolution network and protein language model representation. Journal of Computational Chemistry, 45(8), 436-445. https://doi.org/10.1002/jcc.27249

- Chen, J., Zheng, S., Zhao, H., & Yang, Y. (2021). Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map. Journal of Cheminformatics, 13(1), 7. https://doi.org/10.1186/s13321-021-00488-1

论文中提出的三个关键技术

1. 蛋白质三维结构的图卷积网络表示
   想象蛋白质像一条折叠的绳子，上面有许多珠子(氨基酸)。传统方法只看这串珠子的排列顺序，但忽略了空间上的关系。
   本文的创新是：
   将蛋白质看作一个"图"，其中每个"节点"是一个氨基酸
   用 AlphaFold2(一种先进的 AI 工具)预测蛋白质的三维结构
   根据氨基酸之间的距离建立"邻接矩阵"，记录哪些氨基酸在空间上靠近
   这就像建立了一张蛋白质内部的"关系网络图"，帮助计算机理解蛋白质的空间结构。

2. 使用 ProtTrans 进行序列特征提取
   传统方法中，每个氨基酸只用简单的数字表示。而这篇论文使用了更先进的"蛋白质语言模型"(ProtTrans)：
   这类似于 ChatGPT 这样的语言模型，但专门学习了蛋白质序列的"语言"
   它能够理解氨基酸序列中的复杂模式和上下文关系
   每个氨基酸不再是简单的标签，而是一个包含丰富信息的"向量"
   这就像给每个氨基酸配上了一个详细的"身份证"，包含了它与其他氨基酸的各种关系信息。

3. LSTM 溶解度预测模型
   有了前两部分提取的特征，论文使用 LSTM(一种擅长处理序列数据的神经网络)来进行最终预测：
   结合了空间结构信息和序列特征
   使用 eSOL 数据集(包含约 4132 个大肠杆菌蛋白质)进行训练
   使用酵母菌(S. cerevisiae)数据集进行独立测试
   这就像一个专家在综合考虑了蛋白质的"外形"和"成分"后，给出溶解度变化的预测。
   论文的创新点和意义

初始数据集：使用来自 Pon-Sol2 研究的数据集，包含 77 种蛋白质的 6328 个突变，这些突变已经被标记为三类（溶解度降低、增加或无效）。

数据分割：将这些数据随机分为训练集（5666 个突变）和测试集（662 个突变），同时保持三类突变的比例相似，并确保测试集中的突变位置与训练集不重叠。

【预训练阶段 - LSTM 在这里发挥作用】

- 在 DeepSol 数据集(~69,000 个蛋白质序列)上预训练一个蛋白质溶解度预测模型
- 这个预训练模型使用 LSTM 网络处理 ProtTrans 提取的序列特征
- LSTM 的作用是捕捉序列中的长距离依赖关系
- 预训练模型输出的注意力池化特征将用于后续主模型

三维结构提取：
使用 AlphaFold2 预测这些蛋白质的三维结构（针对野生型蛋白质）
基于预测的三维结构生成邻接矩阵，表示氨基酸之间的空间关系
这些结构信息用于构建蛋白质的图表示（Graph representation）

序列特征提取：
使用 ProtTrans 预训练模型从蛋白质序列中提取特征
对于每个长度为 L 的蛋白质序列，生成 L×1024 维的特征
将野生型和突变型蛋白质的特征拼接，得到 L×2048 维的节点特征

GCN 模型：
将提取的序列特征作为节点特征
使用基于三维结构生成的邻接矩阵表示节点间的边关系
通过图卷积网络聚合蛋白质的空间结构和序列信息

【特征融合 - 使用 LSTM 预训练模型的输出】

- 从预训练的 LSTM 模型中提取 L×8 维的注意力池化特征
- 将这些特征与 GCN 输出的 L×16 维特征拼接

预测：
将 GCN 的输出特征与来自蛋白质溶解度预测模型的特征拼接
通过自注意力池化和全连接层生成最终的溶解度变化预测值

DeepMutSol完整训练流程
阶段一：数据准备和特征提取
数据集收集
Pon-Sol2数据集：包含77种蛋白质的6328个突变
DeepSol数据集：包含~69,000个蛋白质序列(用于预训练)
三维结构预测
使用AlphaFold2预测野生型蛋白质的三维结构
只需预测原始蛋白质结构，不需要预测每个突变后的结构
序列特征提取
使用ProtTrans预训练模型从蛋白质序列中提取特征
对野生型和突变型蛋白质序列都进行特征提取
拼接这些特征形成节点特征


阶段二：预训练溶解度预测模型
LSTM模型构建
使用LSTM网络和自注意力池化模块构建模型
输入为ProtTrans提取的特征
LSTM模型预训练
在DeepSol数据集上训练模型预测蛋白质溶解度
目标是区分可溶性和不可溶性蛋白质
从预训练模型中提取注意力池化特征


阶段三：突变溶解度变化预测模型训练
构建GCN模型
使用预测的三维结构生成邻接矩阵
将ProtTrans特征作为节点特征
通过GCN聚合结构和序列信息
特征融合
将GCN输出特征与预训练LSTM模型的注意力池化特征拼接
通过自注意力池化生成全局嵌入
最终预测
使用全连接层生成最终的溶解度变化预测
在Pon-Sol2数据集上训练和评估模型


我们需要实现的三个关键部分
蛋白质三维结构的GCN邻接矩阵表示 (对应阶段一的步骤2和阶段三的步骤1)
需要的数据：蛋白质PDB文件(来自AlphaFold2预测或DeepMutSol仓库)
实现内容：读取PDB文件，提取Cα原子坐标，计算距离，生成邻接矩阵
代码位置：protein_utils.py
使用ProtTrans进行序列特征提取 (对应阶段一的步骤3)
需要的数据：蛋白质序列数据(从DeepMutSol仓库的xlsx文件中获取)
实现内容：加载ProtTrans模型，输入序列，提取特征，应用池化策略
代码位置：prot_trans.py
LSTM溶解度预测模型 (对应阶段二和阶段三)
需要的数据：
预训练阶段：DeepSol数据集(~69,000个蛋白质序列)
主模型训练：Pon-Sol2数据集(train_dataset.xlsx, test_dataset.xlsx)
前两个步骤生成的特征
实现内容：构建LSTM网络，融合GCN和序列特征，预测溶解度变化
代码位置：solubility_model.py
数据需求总结
PDB文件：蛋白质三维结构数据(DeepMutSol仓库/AlphaFold2)
序列数据：蛋白质序列(DeepMutSol仓库的xlsx文件)
溶解度标签：蛋白质溶解度和突变效应数据(DeepMutSol仓库的xlsx文件)
预训练数据：DeepSol数据集(用于LSTM预训练)
ProtTrans模型：从Hugging Face下载预训练模型
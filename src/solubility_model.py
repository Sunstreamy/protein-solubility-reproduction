import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


class SolubilityPredictor(nn.Module):
    """
    基于LSTM的蛋白质溶解度预测模型

    该模型结合LSTM特征和GCN的图特征来预测蛋白质溶解度
    """

    def __init__(self, input_dim, hidden_dim, gcn_dim, n_layers=2, dropout=0.3):
        super(SolubilityPredictor, self).__init__()

        # 用于序列处理的LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )

        # 注意力机制
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Tanh())

        # 用于溶解度预测的全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2 + gcn_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_seq, x_graph):
        """
        模型的前向传播

        参数:
        -----------
        x_seq : torch.Tensor
            形状为(batch_size, seq_len, input_dim)的序列特征张量
        x_graph : torch.Tensor
            形状为(batch_size, gcn_dim)的来自GCN的图特征张量

        返回:
        --------
        solubility : torch.Tensor
            预测的溶解度值
        """
        # 使用LSTM处理序列
        lstm_out, _ = self.lstm(x_seq)  # (batch_size, seq_len, hidden_dim*2)

        # 应用注意力
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # LSTM输出的加权和
        lstm_features = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch_size, hidden_dim*2)

        # 将LSTM特征与图特征连接
        combined_features = torch.cat([lstm_features, x_graph], dim=1)

        # 预测溶解度
        solubility = self.fc_layers(combined_features)

        return solubility


def train_solubility_model(
    seq_features,
    graph_features,
    solubility_values,
    hidden_dim=128,
    gcn_dim=64,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    test_size=0.2,
    random_state=42,
):
    """
    训练溶解度预测模型

    参数:
    -----------
    seq_features : numpy.ndarray
        来自ProtTrans的序列特征数组(batch_size, seq_len, input_dim)
    graph_features : numpy.ndarray
        来自GCN的图特征数组(batch_size, gcn_dim)
    solubility_values : numpy.ndarray
        实验溶解度值数组
    hidden_dim : int
        LSTM的隐藏维度
    gcn_dim : int
        GCN的图特征维度
    epochs : int
        训练轮数
    batch_size : int
        训练批量大小
    learning_rate : float
        优化器的学习率
    test_size : float
        用于测试的数据比例
    random_state : int
        可重现性的随机种子

    返回:
    --------
    model : SolubilityPredictor
        训练好的模型
    history : dict
        训练历史
    test_metrics : dict
        测试集上的评估指标
    """
    # 将numpy数组转换为PyTorch张量
    seq_features = torch.FloatTensor(seq_features)
    graph_features = torch.FloatTensor(graph_features)
    solubility_values = torch.FloatTensor(solubility_values).reshape(-1, 1)

    # 将数据分为训练集和测试集
    X_seq_train, X_seq_test, X_graph_train, X_graph_test, y_train, y_test = (
        train_test_split(
            seq_features,
            graph_features,
            solubility_values,
            test_size=test_size,
            random_state=random_state,
        )
    )

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_seq_train, X_graph_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 初始化模型
    input_dim = seq_features.shape[2]  # 序列特征的维度
    model = SolubilityPredictor(input_dim, hidden_dim, gcn_dim)

    # 如果可用，将模型移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    history = {"train_loss": [], "train_r2": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []

        for batch_seq, batch_graph, batch_y in train_loader:
            # 将张量移至设备
            batch_seq = batch_seq.to(device)
            batch_graph = batch_graph.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_seq, batch_graph)

            # 计算损失
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 跟踪指标
            epoch_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())

        # 计算每轮指标
        train_loss = epoch_loss / len(train_loader)
        train_r2 = r2_score(all_targets, all_preds)

        # 存储历史
        history["train_loss"].append(train_loss)
        history["train_r2"].append(train_r2)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(
                f"轮次 {epoch+1}/{epochs}, 损失: {train_loss:.4f}, R²: {train_r2:.4f}"
            )

    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        X_seq_test = X_seq_test.to(device)
        X_graph_test = X_graph_test.to(device)
        y_test = y_test.to(device)

        y_pred = model(X_seq_test, X_graph_test)
        test_loss = criterion(y_pred, y_test).item()

        # 将预测结果移回CPU进行评估
        y_pred = y_pred.cpu().numpy()
        y_test = y_test.cpu().numpy()

        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    test_metrics = {"loss": test_loss, "r2": test_r2, "rmse": test_rmse}

    print(
        f"测试损失: {test_loss:.4f}, 测试R²: {test_r2:.4f}, 测试RMSE: {test_rmse:.4f}"
    )

    return model, history, test_metrics

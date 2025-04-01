import torch
import numpy as np


def extract_features_with_prottrans(
    sequences, model_name="Rostlab/prot_bert", pooling_strategy="mean"
):
    """
    使用ProtTrans模型提取蛋白质序列特征

    参数:
    -----------
    sequences : list
        蛋白质序列字符串列表
    model_name : str
        要使用的ProtTrans模型名称。选项包括:
        - "Rostlab/prot_bert" (默认)
        - "Rostlab/prot_t5_xl_uniref50"
        - "Rostlab/prot_electra_discriminator"
        - "Rostlab/prot_albert"
    pooling_strategy : str
        将标记嵌入汇总为序列嵌入的策略
        选项: "mean", "cls", "attention"

    返回:
    --------
    embeddings : numpy.ndarray
        蛋白质嵌入数组
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("请安装transformers库: pip install transformers")
        return None

    # 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 如果可用，将模型移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_embeddings = []

    # 批量处理序列
    batch_size = 4  # 根据GPU内存调整
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i : i + batch_size]

        # 对序列进行分词
        inputs = tokenizer(
            batch_sequences, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 提取特征
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取隐藏状态
        hidden_states = outputs.last_hidden_state

        # 应用池化策略
        batch_embeddings = []
        for j, seq in enumerate(batch_sequences):
            # 获取当前序列的注意力掩码
            attention_mask = inputs["attention_mask"][j]

            if pooling_strategy == "mean":
                # 平均池化：对所有标记嵌入求平均，忽略填充
                masked_embeddings = hidden_states[j] * attention_mask.unsqueeze(-1)
                sum_embeddings = masked_embeddings.sum(dim=0)
                seq_len = attention_mask.sum().item()
                embedding = sum_embeddings / seq_len

            elif pooling_strategy == "cls":
                # CLS池化：使用[CLS]标记嵌入
                embedding = hidden_states[j][0]

            elif pooling_strategy == "attention":
                # 注意力池化：基于注意力的加权平均
                # 这是一个简化版本
                attention_weights = torch.softmax(attention_mask.float(), dim=0)
                embedding = (hidden_states[j] * attention_weights.unsqueeze(-1)).sum(
                    dim=0
                )

            batch_embeddings.append(embedding.cpu().numpy())

        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)

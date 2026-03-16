import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import GeometryTokenizer

class GeometryDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 128):
        """
        jsonl_path: 数据集路径
        tokenizer: 我们刚才实例化的分词器
        max_length: 序列的最大长度。不足会补齐，超出会截断。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 将所有 JSONL 数据加载到内存中
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        """数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """取出第 idx 条数据，并转化为张量"""
        sample = self.data[idx]
        premises = sample["premises"]
        target = sample["target"]
        aux_points = sample["auxiliary_points"]

        # 调用 tokenizer 将文本转为数字 ID
        input_ids = self.tokenizer.encode(premises, target, aux_points)

        # 构建 Labels，只对要预测的部分计算 Loss
        labels = input_ids.copy()
        
        # 寻找第二个 <sep> 的位置
        sep_id = self.tokenizer.vocab["<sep>"]
        sep_indices = [i for i, token_id in enumerate(input_ids) if token_id == sep_id]

        if len(sep_indices) >= 2:
            # 第二个 <sep> 及其之前的所有内容设为 -100
            prompt_end_idx = sep_indices[1]
            for i in range(prompt_end_idx + 1):
                labels[i] = -100
        else:
            # 如果数据格式异常，则整条不计算 loss
            labels = [-100] * len(labels)

        # 对齐长度
        pad_id = self.tokenizer.vocab["<pad>"]

        if len(input_ids) < self.max_length:
            # 不足 max_length，用 <pad> 补齐
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([pad_id] * padding_length)
            labels.extend([-100] * padding_length)
        else:
            # 超出 max_length 则截断
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # 转化为 PyTorch Tensor 返回
        return {
            "input_ids": torch.tensor(input_ids, dtype = torch.long),
            "labels": torch.tensor(labels, dtype = torch.long)
        }


if __name__ == "__main__":
    # 初始化 Tokenizer 并建表
    tokenizer = GeometryTokenizer()
    data_path = "traindata/synthetic_data.jsonl"
    tokenizer.build_vocab_from_jsonl(data_path)

    # 实例化 Dataset
    dataset = GeometryDataset(data_path, tokenizer, max_length=64)
    print(f"\n成功加载数据集，共 {len(dataset)} 条数据。")

    # 实例化 DataLoader (将数据打包成批次)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

    # 抓取一个 Batch 并进行输出查看
    for batch in dataloader:
        print("\n=== 查看一个 Batch 的数据结构 ===")
        print("Input IDs shape:", batch["input_ids"].shape)
        print("Labels shape:", batch["labels"].shape)
        
        # 打印第一条数据的 input_ids 和 labels
        print("\n第一条数据的 Input IDs:\n", batch["input_ids"][0].tolist())
        print("第一条数据的 Labels (-100 表示不计算 Loss):\n", batch["labels"][0].tolist())
        break

    
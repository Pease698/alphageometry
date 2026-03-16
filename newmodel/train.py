import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tokenizer import GeometryTokenizer
from dataset import GeometryDataset
from model import create_mini_geometry_model

def train():
    # 硬件探测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 准备基础组件
    data_path = "traindata/synthetic_data.jsonl"
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl(data_path)
    vocab_size = len(tokenizer.vocab)
    pad_id = tokenizer.vocab["<pad>"]

    dataset = GeometryDataset(data_path, tokenizer, max_length = 128)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

    # 实例化模型并移动到计算设备上
    model = create_mini_geometry_model(vocab_size)
    model.to(device)

    # 定义优化器 AdamW，学习率为 5e-4
    optimizer = AdamW(model.parameters(), lr = 5e-4)

    # ================= 开始训练 =================
    num_epochs = 30 # 训练轮数
    model.train()   # 将模型设置为训练模式

    print("\n" + "="*40)
    print("开始训练 (Training Loop)")
    print("="*40)

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # 遍历数据加载器中的每一个批次
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 动态生成 attention_mask
            # 非 <pad> 标记为 1 表示需要关注，<pad> 标记为 0 表示忽略
            attention_mask = (input_ids != pad_id).long().to(device)

            # 清空旧梯度
            optimizer.zero_grad()

            # 前向传播，得出 Loss
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss 

            # 反向传播，计算新梯度
            loss.backward()

            # 更新模型权重
            optimizer.step()

            # 累加这个批次的 Loss
            total_loss += loss.item()

        # 打印当前 Epoch 的平均 Loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Average Loss: {avg_loss:.4f}")

    # 训练结束，保存模型成果
    save_dir = "./mini_ag_weights"
    print(f"\n训练大功告成，正在保存至 {save_dir} ...")
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    train()


import torch
from transformers import GPT2LMHeadModel
from tokenizer import GeometryTokenizer

def load_model_and_tokenizer():
    """加载训练好的模型和分词器密码本"""
    # 重新构建相同的词表
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl("traindata/synthetic_data.jsonl") 
    
    # 从刚才保存的文件夹加载权重
    save_dir = "./mini_ag_weights"
    model = GPT2LMHeadModel.from_pretrained(save_dir)
    
    # 将模型设置为评估模式
    model.eval() 
    return model, tokenizer

def generate_auxiliary_points(model, tokenizer, premises, target, device = "cpu"):
    """给定前提和目标，让模型生成辅助点"""
    # 组装 Prompt，只拼接到第二个 <sep>
    premises_str = " ; ".join(premises)
    prompt = f"<bos> {premises_str} <sep> {target} <sep>"
    
    # 将文本 Prompt 转化为数字 ID
    input_ids = []
    for token in prompt.split():
        # 使用 get()，未知字符默认转为 <pad> (ID: 0)
        input_ids.append(tokenizer.vocab.get(token, 0)) 
        
    # 转化为 PyTorch 张量，并增加 Batch 维度 (1, sequence_length)
    input_tensor = torch.tensor([input_ids], dtype = torch.long).to(device)
    
    print(f"\n输入给模型的 Prompt:\n{prompt}")
    print("-" * 30)
    
    # 开始生成
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens = 20,
            pad_token_id = tokenizer.vocab["<pad>"],
            eos_token_id = tokenizer.vocab["<eos>"],
            do_sample = False,       # 每次选择概率最大的词
            # temperature=0.7
        )
        
    # 解码输出，将 prompt 删除
    generated_ids = output_ids[0][len(input_ids):].tolist()
    
    # 若生成了 <eos> 则删除后续内容
    if tokenizer.vocab["<eos>"] in generated_ids:
        eos_index = generated_ids.index(tokenizer.vocab["<eos>"])
        generated_ids = generated_ids[:eos_index]
        
    # 翻译回人类可读的几何指令
    predicted_aux = tokenizer.decode(generated_ids)
    return predicted_aux

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    
    # 测试题目
    # test_premises = ["D d a d b", "D d c d b", "^ c d c b b c b d", "T a e b c", "C f c b", "T c b f d", "P h b d f", "O a e k c"]
    # test_target = "^ e k b d c k a b"

    test_premises = ["T e f a c", "C a c f", "T b h d e"]
    test_target = "^ b h e f d e a c"
    
    print("\n引擎启动，开始推理...")
    prediction = generate_auxiliary_points(model, tokenizer, test_premises, test_target, device)
    
    print(f"模型预测的辅助点:  {prediction}")
    # print(f"数据集里的标准答案: C a l b")
    print(f"数据集里的标准答案: P g d c f ; T c a n i")


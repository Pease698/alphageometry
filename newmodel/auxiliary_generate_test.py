import torch
import json
import random
from transformers import GPT2LMHeadModel
from tokenizer import GeometryTokenizer

def load_model_and_tokenizer():
    """加载训练好的模型和分词器密码本"""
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl("traindata/synthetic_data_v3.jsonl") 
    
    save_dir = "./mini_ag_weights"
    model = GPT2LMHeadModel.from_pretrained(save_dir)
    model.eval() 
    return model, tokenizer

def generate_multiple_auxiliary_points(model, tokenizer, premises, target, num_proposals=3, device="cpu", verbose=False):
    """给定前提和目标，使用束搜索(Beam Search)让模型生成多个辅助点方案"""
    premises_str = " ; ".join(premises)
    prompt = f"<bos> {premises_str} <sep> {target} <sep>"
    
    input_ids = []
    for token in prompt.split():
        input_ids.append(tokenizer.vocab.get(token, 0)) 
        
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    if verbose:
        print(f"输入 Prompt: {prompt}")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=50,
            pad_token_id=tokenizer.vocab.get("<pad>", 0),
            eos_token_id=tokenizer.vocab.get("<eos>", 0),
            do_sample=False,              # 关闭随机采样，保证结果的确定性和逻辑性
            num_beams=num_proposals,      # 束搜索的宽度
            num_return_sequences=num_proposals, # 返回多少个序列
            early_stopping=True           # 遇到 eos token 提前停止
        )
        
    predictions = []
    # 遍历生成的多个方案
    for i in range(num_proposals):
        # 截取新生成的 token 部分
        generated_ids = output_ids[i][len(input_ids):].tolist()
        
        # 截断到 <eos> 为止
        eos_id = tokenizer.vocab.get("<eos>")
        if eos_id in generated_ids:
            eos_index = generated_ids.index(eos_id)
            generated_ids = generated_ids[:eos_index]
            
        predicted_aux = tokenizer.decode(generated_ids)
        predictions.append(predicted_aux.strip())
        
    return predictions

def evaluate_random_samples(model, tokenizer, dataset_path, num_samples=10, num_proposals=3, device="cpu"):
    """从数据集中随机抽取样本进行多方案测试并对比结果"""
    print(f"\n正在从 {dataset_path} 中加载并打乱数据...")
    
    all_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line.strip()))
                
    sampled_data = random.sample(all_data, min(num_samples, len(all_data)))
    
    print(f"成功抽取 {len(sampled_data)} 条测试题。开始闭卷考试 (Top-{num_proposals} 评估)：\n")
    
    top_k_correct_count = 0
    
    for i, data in enumerate(sampled_data, 1):
        problem_id = data.get("problem_id", f"Unknown_{i}")
        premises = data.get("premises", [])
        target = data.get("target", "")
        
        ground_truth_list = data.get("auxiliary_points", [])
        ground_truth = " ; ".join(ground_truth_list)
        
        print(f"{'='*15} 第 {i} 题 (ID: {problem_id}) {'='*15}")
        print(f"标准答案 : {ground_truth}\n")
        
        # 【核心修改区】：调用新函数获取多个预测结果
        predictions = generate_multiple_auxiliary_points(
            model, tokenizer, premises, target, num_proposals=num_proposals, device=device, verbose=False
        )
        
        match_found = False
        for j, pred in enumerate(predictions, 1):
            if pred == ground_truth:
                print(f"  方案 {j} : {pred}  ✅ (命中)")
                match_found = True
            else:
                print(f"  方案 {j} : {pred}  ❌")
                
        if match_found:
            print(f"\n综合评估 : 🎉 该题 Pass@{num_proposals} 测试通过！")
            top_k_correct_count += 1
        else:
            print(f"\n综合评估 : 😔 该题的所有方案均未命中标准答案。")
        print("\n")
        
    print(f"{'*'*40}")
    print(f"测试总结：在 {len(sampled_data)} 道题目中，有 {top_k_correct_count} 道题目的 Top-{num_proposals} 预测中包含了标准答案。")
    print(f"Top-{num_proposals} 准确率：{(top_k_correct_count / len(sampled_data)) * 100:.2f}%")
    print(f"{'*'*40}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    
    test_dataset_path = "traindata/synthetic_data_v3.jsonl" 
    
    # 你可以在这里调整 num_proposals 的数量，比如设置为 3 或 5
    evaluate_random_samples(model, tokenizer, test_dataset_path, num_samples=20, num_proposals=10, device=device)


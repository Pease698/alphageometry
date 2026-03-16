import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

class GeometryTokenizer:
    def __init__(self):
        # 特殊字符
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<sep>", ";"]
        
        # 初始化词表字典和反向词表
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 特殊字符加入词表
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token):
        """将单个字符加入词表"""
        if token not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
            
    def build_vocab_from_jsonl(self, file_path):
        """遍历整个数据集，收集所有出现过的字符"""
        print("开始构建词表...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # 提取所有文本部分
                all_text_parts = []
                all_text_parts.extend(data["premises"])
                all_text_parts.append(data["target"])
                all_text_parts.extend(data["auxiliary_points"])
                
                # 将文本打散成单个 token 并加入词表
                for text in all_text_parts:
                    tokens = text.split() 
                    for token in tokens:
                        self.add_token(token)
                        
        print(f"词表构建完成！共包含 {len(self.vocab)} 个独立的 Token。")

    def encode(self, premises, target, aux_points):
        """将一条数据转化为数字 ID 列表"""
        # 拼接成完整的字符串序列
        premises_str = " ; ".join(premises)
        aux_str = " ; ".join(aux_points)
        full_sequence = f"<bos> {premises_str} <sep> {target} <sep> {aux_str} <eos>"
        
        # 切分成 token 列表
        tokens = full_sequence.split()
        
        # 转化为数字 ID
        # 如果遇到词表里没有的词，默认回退到 <pad>
        token_ids = [self.vocab.get(token, 0) for token in tokens] 
        return token_ids

    def decode(self, token_ids):
        """将数字 ID 还原为可读文本（用于验证和推理阶段）"""
        tokens = [self.inverse_vocab.get(tid, "<unk>") for tid in token_ids]
        return " ".join(tokens)


if __name__ == "__main__":
    # 实例化分词器
    tokenizer = GeometryTokenizer()
    
    # 假设你的文件名为 synthetic_data.jsonl
    data_path = "traindata/synthetic_data.jsonl" 
    
    # 扫描数据，构建词表
    tokenizer.build_vocab_from_jsonl(data_path)
    
    # 打印看看词表长什么样
    print("\n完整词表映射:", tokenizer.vocab)
    
    # 拿第一条数据测试一下编码功能
    test_premises = ["T e f a c", "C a c f", "T b h d e"]
    test_target = "^ b h e f d e a c"
    test_aux = ["P g d c f", "T c a n i"]
    
    encoded_ids = tokenizer.encode(test_premises, test_target, test_aux)
    print("\n原始数据编码后的数字 ID:", encoded_ids)
    
    # 测试解码功能，看看能不能精准还原
    decoded_text = tokenizer.decode(encoded_ids)
    print("\n数字 ID 解码还原的文本:", decoded_text)


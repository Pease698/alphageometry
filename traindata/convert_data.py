import json

def extract_points(text):
    """提取字符串中所有的单小写字母（代表几何点）"""
    # 比如 "P g d c f" 会返回 ['g', 'd', 'c', 'f']
    return [tok for tok in text.split() if len(tok) == 1 and tok.islower()]

def convert_dataset(input_file, output_file):
    print(f"🔄 开始读取并转换数据: {input_file}")
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line.strip())
            
            # 1. 收集这道题最初的“已知点”
            known_points = set()
            for prem in data['premises']:
                known_points.update(extract_points(prem))
                
            # 2. 转换辅助点格式
            new_aux_points = []
            for aux in data['auxiliary_points']:
                pts_in_aux = extract_points(aux)
                
                # 寻找“生面孔”：找出第一个不在 known_points 中的点，作为“新点”
                new_point = None
                for p in pts_in_aux:
                    if p not in known_points:
                        new_point = p
                        break
                
                # 容错处理：理论上辅助线一定会引入新点，如果全都是老点，默认取第一个
                if not new_point and pts_in_aux:
                    new_point = pts_in_aux[0]
                elif not new_point:
                    new_point = "x" 
                    
                # 3. 核心改造：拼接成 DeepMind 官方格式 "点名 : 原始指令"
                formatted_aux = f"{new_point} : {aux}"
                new_aux_points.append(formatted_aux)
                
                # 更新“已知点”集合，供下一条辅助线使用
                known_points.update(pts_in_aux)
                
            # 将改造后的辅助点塞回数据字典
            data['auxiliary_points'] = new_aux_points
            
            # 写入新文件
            fout.write(json.dumps(data) + '\n')
            processed_count += 1
            
    print(f"✅ 转换大功告成！共处理 {processed_count} 条数据。")
    print(f"📁 新数据已保存至: {output_file}")

if __name__ == "__main__":
    # 请确保路径与你实际的文件路径一致
    INPUT_PATH = "traindata/synthetic_data.jsonl"
    OUTPUT_PATH = "traindata/synthetic_data_v2.jsonl"
    
    convert_dataset(INPUT_PATH, OUTPUT_PATH)


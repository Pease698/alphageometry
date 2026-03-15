import os
import sys
import random
import json

# 动态配置系统路径，确保能无缝调用 reuse 里的模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'reuse'))
sys.path.append(os.path.join(PROJECT_ROOT, 'origin'))
sys.path.append(os.path.join(PROJECT_ROOT, 'meliad_lib', 'meliad'))

from reuse import problem
from reuse import graph
from reuse import ddar
from reuse import trace_back
from reuse import pretty


def generate_random_premises(num_extra_clauses=3):
    """
    随机前提生成器
    :param num_extra_clauses: 除了基础三角形外，想要额外添加的随机前提数量
    """
    points = ['a', 'b', 'c']
    premises_str = "a b c = triangle"
    
    # 准备英文字母表，用于给新产生的点命名
    available_names = [chr(i) for i in range(100, 123)] # d 到 z
    
    # 定义辅助作图动作池
    # 格式：(动作名称, 该动作需要的已知点数量)
    safe_actions = [
        ("midpoint", 2),   # 取两点连线的中点
        ("on_line", 2),    # 在两点连线上任取一点
        ("on_bline", 2),   # 在两点中垂线上任取一点
        ("foot", 3),       # 作一点到另外两点连线的垂足
        ("on_pline", 3),   # 过一点作另外两点连线的平行线，并在上面任取一点
        ("on_tline", 3),   # 过一点作另外两点连线的垂线，并在上面任取一点
    ]
    
    # 循环生成随机作图步骤
    for _ in range(num_extra_clauses):
        if not available_names:
            break # 字母用完则停止
            
        # 随机挑选一个作图动作
        action, required_points = random.choice(safe_actions)
        
        # 从已经存在的点中，随机抽取不重复的点作为动作的输入
        sampled_points = random.sample(points, required_points)
        
        # 取出一个新字母作为新点的名字
        new_p = available_names.pop(0)
        
        # 语法格式："新点 = 动作 新点 已知点1 已知点2 ..."
        if required_points == 2:
            clause = f"{new_p} = {action} {new_p} {sampled_points[0]} {sampled_points[1]}"
        elif required_points == 3:
            clause = f"{new_p} = {action} {new_p} {sampled_points[0]} {sampled_points[1]} {sampled_points[2]}"
        
        # 将新条件追加到总字符串中
        premises_str += f"; {clause}"
        
        # 将新点加入已有的“点池”
        points.append(new_p)
        
    return f"synthetic_random_graph\n{premises_str}"


def format_dep(dep, refs):
    """
    【新增】格式化工具函数：将 Dependency 对象转化为带引用编号的自然语言
    """
    # 解析参数名，处理可能仍是 Point 对象的参数
    args_str = [arg.name if hasattr(arg, 'name') else str(arg) for arg in dep.args]
    # 调用 pretty 转化为自然语言
    nl = pretty.pretty_nl(dep.name, args_str) or pretty.pretty([dep.name] + args_str)
    
    # 尝试从全局 refs 字典中获取该定理的编号
    h = dep.hashed()
    if h in refs:
        return f"{nl} [{refs[h]:02d}]"
    return nl

def generate_data(num_extra_clauses = 5, file_path = None):
    """
    生成合成几何数据
    :param num_extra_clauses: 额外添加的随机前提数量
    :param file_path: 可选参数，指定将生成的数据保存到哪个文件
    """
    output_flag = True
    if file_path is not None:
        output_flag = False

    if output_flag:
        print("=== 启动 AlphaGeometry 微型数据生成器 ===")

    # 加载几何定义和推理规则
    defs_path = os.path.join(PROJECT_ROOT, 'defs.txt')
    rules_path = os.path.join(PROJECT_ROOT, 'rules.txt')

    # 导入几何定义和推导规则
    definitions = problem.Definition.from_txt_file(defs_path, to_dict=True)
    theorems = problem.Theorem.from_txt_file(rules_path, to_dict=True)

    generation_attempts = 0 # 推导前提随机次数

    while True:
        generation_attempts += 1
        if output_flag:
            print(f"\n=== 第 {generation_attempts} 次随机前提生成尝试 ===")

        # 模拟“随机采样前提”，随机连续作图若干次
        simulated_random_premises = generate_random_premises(num_extra_clauses)
        if output_flag:
            print(f"[*] 初始随机字符串:\n{simulated_random_premises}")

        # 使用 Problem 类解析
        p = problem.Problem.from_txt(simulated_random_premises, translate=True)
    
        if output_flag:
            print(f"\n[*] 解析成功！")

            # 初始化证明状态图 (Graph)
            print("\n[*] 正在根据 Definitions 搭建证明状态图...")

        try:
            g, added_deps = graph.Graph.build_problem(p, definitions, verbose=False)
        except ValueError as e:
            if output_flag:
                print(f"[-] 运气不佳，生成了退化图形 ({e})，正在重新生成...")
            continue

        if output_flag:
            print(f"[*] 建图成功！当前图内已有 {len(g.all_nodes())} 个初始节点。")

            # 符号引擎 (DD+AR) 推导，生成 DAG
            print("[*] 正在运行 DD+AR 引擎展开演绎闭包...")

        # 按照 ddar.solve 的真实签名传参
        # g: 状态图, theorems: 规则列表, controller: 问题对象 p
        # 因为生成数据时 p.goal 是 None，所以它会一直运行到 max_level 或饱和为止
        g, level_times, status, branches, all_added = ddar.solve(
            g, 
            theorems, 
            controller=p, 
            max_level=1000, 
            timeout=600
        )

        if output_flag:
            print(f"[*] 推导完成！图中当前共有 {len(g.all_nodes())} 个几何结论节点。")

        if not all_added:
            if output_flag:
                print("[-] 初始条件太简单")
            return

        target_dep = all_added[-1]
        
        if output_flag:
            print("\n[*] 正在启动 Traceback 回溯算法，提取最短证明树...")
        
        # 提取原始逻辑链
        setup_raw, aux_raw, log_raw, setup_points = trace_back.get_logs(target_dep, g, merge_trivials=True)

        if file_path is not None and len(aux_raw) == 0:
            print("[-] 没有辅助点，重新生成")
            continue # 如果没有辅助构造点，则重新生成

        if file_path is not None:
            # 提取 DSL 格式的输入 (Input)
            # 使用 pretty.pretty 函数将 Dependency 对象转回类似 "T a b c d" 的底层符号串
            setup_dsl = [pretty.pretty([dep.name] + [a.name if hasattr(a, 'name') else str(a) for a in dep.args]) for dep in setup_raw]
            
            target_args = [a.name if hasattr(a, 'name') else str(a) for a in target_dep.args]
            target_dsl = pretty.pretty([target_dep.name] + target_args)

            # 提取 DSL 格式的标签 (Label - 辅助点)
            aux_dsl = [pretty.pretty([dep.name] + [a.name if hasattr(a, 'name') else str(a) for a in dep.args]) for dep in aux_raw]

            # 构建训练样本字典
            training_sample = {
                "problem_id": f"synthetic_{random.randint(10000, 99999)}",
                "premises": setup_dsl,        # 模型输入 1
                "target": target_dsl,         # 模型输入 2
                "auxiliary_points": aux_dsl,  # 模型预测标签
                "proof_length": len(log_raw)  # 元数据：可用于后续按难度过滤数据
            }

            #以追加模式 ('a') 写入 JSONL 文件
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                # 写入成功后退出循环
                break 
            except Exception as e:
                print(f"写入文件失败: {e}")
                break
        else:
            # 初始化全局引用字典 (Reference ID Dictionary)，将所有 Dependency 对象的 Hash 值映射为一个整数 ID (例如 00, 01)
            refs = {} 

            # 分发引用编号并按点分组。point_log 会将孤立的已知条件，按照它们界定了哪个“点”来进行打包，同时给每个事实分配 refs
            setup_grouped = trace_back.point_log(setup_raw, refs, set())
            aux_grouped = trace_back.point_log(aux_raw, refs, setup_points)

            if output_flag:
                print("\n========================================================")
                print("合成几何题生成完毕 (Synthetic Data Ready)")
                print("========================================================")
    
                print("\n * From theorem premises (精简初始条件):")

            # 遍历打包后的初始条件
            for points, cons in setup_grouped:
                pts_names = ",".join(p.name.upper() for p in points)
                if output_flag:
                    print(f"  {pts_names} : Points")
                    for dep in cons:
                        print(f"  {format_dep(dep, refs)}")

            if output_flag:
                print("\n * Auxiliary Constructions (辅助构造点):")
            if not aux_grouped:
                if output_flag:
                    print("  (None)")
            else:
                for points, cons in aux_grouped:
                    pts_names = ",".join(p.name.upper() for p in points)
                    if output_flag:
                        print(f"  {pts_names} : Points")
                        for dep in cons:
                            print(f"  {format_dep(dep, refs)}")

            # 处理目标结论的自然语言表达
            t_args = [arg.name if hasattr(arg, 'name') else str(arg) for arg in target_dep.args]
            target_str = pretty.pretty_nl(target_dep.name, t_args) or pretty.pretty([target_dep.name] + t_args)
            if output_flag:
                print(f"\n * Target Conclusion (求证目标):\n  ? {target_str}")

                print("\n * Proof steps (精简证明过程):")

            for step_idx, (premises, conclusions) in enumerate(log_raw):
                # 获取前提的字符串表示 (带着引用编号)
                prem_strs = [format_dep(p, refs) for p in premises]

                # 对于当前步骤推导出的新结论，我们需要为它们注册新的 refs 编号
                for c in conclusions:
                    if c.hashed() not in refs:
                        refs[c.hashed()] = len(refs) # 分配新编号

                    c_str = format_dep(c, refs)
                    prem_text = " & ".join(prem_strs) if prem_strs else "已知"

                    # 尝试翻译规则名，如果是空说明是代数替换或合并等平凡步骤
                    rule = f"({c.rule_name})" if c.rule_name else ""

                    # 打印出标准格式： 001. 前提1 [01] & 前提2 [05] (定理) => 结论 [06]
                    if output_flag:
                        print(f"  {step_idx + 1:03d}. {prem_text} {rule} ⇒  {c_str}")

            if output_flag:
                print("========================================================")
            break


def main():
    file_path = os.path.join(PROJECT_ROOT, 'synthetic_data.jsonl')
    generate_data(20, file_path)


if __name__ == "__main__":
    main()


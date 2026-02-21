import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os
import math
import jieba
from sentence_transformers import SentenceTransformer, util

def extract_requirement(line: str) -> Tuple[str, str]:
    """从一行文本中提取需求类型和内容"""
    line = line.strip()
    if not line:
        return "", ""
    
    # 匹配【需求类型】：内容 格式
    pattern = r'【(.+?)】：(.+)'
    match = re.search(pattern, line)
    
    if match:
        req_type = match.group(1).strip()
        req_content = match.group(2).strip()
        return req_type, req_content
    
    # 如果没有匹配到格式，尝试其他方式
    for sep in ['：', ':', '】', ']', ')', '>']:
        if sep in line:
            parts = line.split(sep, 1)
            if len(parts) == 2:
                req_type = parts[0].replace('【', '').replace('[', '').replace('(', '').replace('<', '').strip()
                req_content = parts[1].strip()
                return req_type, req_content
    
    # 如果都没有匹配，返回原始文本作为内容
    return "", line


class OptimizedRequirementEvaluator:
    """优化的需求评估器 - 使用SentenceTransformer批量编码"""
    
    def __init__(self, model_path: str):
        """
        初始化评估器
        
        Args:
            model_path: BGE模型路径
        """
        print(f"正在加载SentenceTransformer模型: {model_path}")
        self.model = SentenceTransformer(model_path)
        print(f"模型加载完成")
        
        # 模糊词字典（权重1-3，3表示最模糊）
        # 只包含真正的模糊词，移除合理的业务词汇
        self.ambiguous_words = {
            # 高权重：强烈不确定性
            '大概': 3, '可能': 3, '也许': 3, '或许': 3, '似乎': 3,
            '大约': 3, '左右': 3, '差不多': 3, '基本上': 3,
            
            # 中权重：中等不确定性  
            '通常': 2, '一般': 2, '常常': 2, '经常': 2, '偶尔': 2,
            '基本上': 2, '主要': 2, '相对': 2, '适当': 2,
            
            # 低权重：轻微不确定性
            '尽量': 1, '尽可能': 1, '争取': 1, '考虑': 1, '酌情': 1,
            '视情况': 1, '相关': 1, '某些': 1, '部分': 1
        }
        
        # 可验证特征模式
        self.verification_patterns = [
            (r'(\d+\.?\d*)\s*(秒|毫秒|ms|分钟|小时|天|%|个|次|倍)', 2),
            (r'(?:响应时间|处理时间|加载时间).*?[≤≥><=]?\s*(\d+)', 2),
            (r'(?:成功率|准确率|可用率).*?([≥>]?)\s*(\d+)%', 2),
            (r'(?:大于|小于|等于|不低于|不超过|超过|少于)\s*(\d+)', 1.5),
            (r'[≤≥><=]\s*(\d+)', 1.5),
            (r'支持(?:.*?)的?(查询|导出|导入|计算|验证|搜索|查看|下载)', 1),
            (r'提供(?:.*?)的?(功能|接口|API|报告|列表|页面)', 1),
            (r'确保(?:.*?)的?(正确|完整|一致|成功|准确|安全)', 1),
        ]
    
    def batch_encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        if not texts:
            return np.array([])
        
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    
    def compute_similarity_matrix(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """批量计算相似度矩阵"""
        if not texts1 or not texts2:
            return np.array([])
        
        embeddings1 = self.batch_encode_texts(texts1)
        embeddings2 = self.batch_encode_texts(texts2)
        
        # 计算相似度矩阵
        similarity_matrix = util.cos_sim(embeddings1, embeddings2)
        
        return similarity_matrix.cpu().numpy()
    
    def evaluate_unambiguity(self, text: str) -> float:
        """
        评估明确性 (0-1分) - 使用jieba分词
        改进版：更合理的模糊词处理和评分逻辑
        """
        if not text.strip():
            return 0.0
        
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 明确的软件和业务术语 - 不应该被标记为模糊
        clear_terms = {
            # 软件功能词汇
            '可视化', '接口', 'API', '模板', '仪表盘', '引擎', '工具',
            '功能', '模块', '系统', '平台', '服务', '组件',
            
            # 操作词汇
            '上传', '下载', '导出', '导入', '同步', '备份', '恢复',
            '查询', '筛选', '排序', '统计', '计算', '验证', '审核',
            
            # 质量属性词汇（有上下文时是明确的）
            '实时', '流畅', '稳定', '高效', '快速', '便捷', '灵活',
            '友好', '清晰', '详细', '智能', '强大', '完善', '安全',
            '可靠', '准确', '完整', '一致', '成功',
            
            # 文件格式
            'Excel', 'PDF', 'CSV', 'JSON', 'XML', 'Word',
            
            # 技术标准
            'MySQL', 'Chrome', 'Firefox', 'Edge', 'Safari',
            'iOS', 'Android', '微信小程序', '移动端', '浏览器',
            
            # 教育管理特定术语
            '选课', '课程', '课表', '学分', '先修课', '教学大纲',
            '成绩', '考试', '作业', '教师', '学生', '管理员',
            '教务', '院系', '专业', '年级', '班级', '容量'
        }
        
        # 检测真正的模糊词
        ambiguous_count = 0
        for word in words:
            # 跳过明确的术语
            if word in clear_terms:
                continue
                
            # 检查是否在模糊词字典中
            if word in self.ambiguous_words:
                ambiguous_count += 1
        
        total_words = len(words)
        if total_words == 0:
            return 1.0
        
        # 更合理的评分逻辑
        ambiguous_ratio = ambiguous_count / total_words
        
        # 根据模糊词比例给出分数
        if ambiguous_ratio == 0:
            return 1.0
        elif ambiguous_ratio <= 0.01:  # 1%以下
            return 0.99
        elif ambiguous_ratio <= 0.02:  # 2%以下
            return 0.98
        elif ambiguous_ratio <= 0.05:  # 5%以下
            return 0.95
        else:
            # 超过5%才显著扣分，但有下限
            return max(0.7, 1.0 - ambiguous_ratio * 5)
    
    def evaluate_verifiability(self, text: str) -> float:
        """
        评估可验证性 (0-1分)
        """
        if not text.strip():
            return 0.0
        
        total_features = 0
        for pattern, weight in self.verification_patterns:
            matches = re.findall(pattern, text)
            if matches:
                total_features += len(matches) * weight
        
        # 使用S型函数将特征数映射到0-1分
        k = 1.5
        c = 2
        
        if total_features == 0:
            return 0.0
        
        verifiability_score = 1 / (1 + math.exp(-k * (total_features - c)))
        
        return round(verifiability_score, 3)
    
    def evaluate_all_metrics(self, golden_reqs: List[Dict], generated_reqs: List[Tuple[str, str]]) -> Dict:
        """
        评估生成需求的所有指标：一致性、完全性、明确性、可验证性
        使用批量计算优化性能
        """
        if not golden_reqs or not generated_reqs:
            return {
                "consistency": 0.0,
                "completeness": 0.0,
                "unambiguity": 0.0,
                "verifiability": 0.0,
                "matched_count": 0,
                "total_golden": len(golden_reqs),
                "match_rate": 0.0,
                "total_generated": len(generated_reqs)
            }
        
        # 提取文本内容
        golden_texts = [req.get("req_description", "") for req in golden_reqs]
        golden_types = [req.get("req_type", "") for req in golden_reqs]
        
        generated_texts = [content for _, content in generated_reqs]
        generated_types = [req_type for req_type, _ in generated_reqs]
        
        # 1. 批量计算相似度矩阵（优化关键步骤）
        print("  批量计算相似度矩阵...")
        similarity_matrix = self.compute_similarity_matrix(golden_texts, generated_texts)
        
        # 2. 计算一致性和完全性
        matched_count = 0
        total_similarity = 0.0
        
        for i, (gold_type, gold_text) in enumerate(zip(golden_types, golden_texts)):
            if not gold_text.strip():
                continue
            
            best_similarity = 0.0
            
            for j, (gen_type, gen_text) in enumerate(zip(generated_types, generated_texts)):
                if not gen_text.strip():
                    continue
                
                # 需求类型必须完全一致
                if gold_type != gen_type:
                    continue
                
                # 从相似度矩阵中获取相似度
                similarity = similarity_matrix[i][j]
                
                if similarity > best_similarity:
                    best_similarity = similarity
            
            # 如果最佳相似度≥0.7，则认为匹配
            if best_similarity >= 0.7:
                matched_count += 1
                total_similarity += best_similarity
        
        # 计算一致性和完全性指标
        total_golden = len(golden_reqs)
        
        # 完全性 = 匹配的需求数 / 黄金需求总数
        completeness = matched_count / total_golden if total_golden > 0 else 0.0
        
        # 一致性 = 所有匹配需求的平均相似度
        consistency = total_similarity / matched_count if matched_count > 0 else 0.0
        
        # 3. 计算明确性和可验证性指标
        print("  计算明确性和可验证性...")
        unambiguity_scores = []
        verifiability_scores = []
        
        for _, req_content in generated_reqs:
            unambiguity_score = self.evaluate_unambiguity(req_content)
            verifiability_score = self.evaluate_verifiability(req_content)
            
            unambiguity_scores.append(unambiguity_score)
            verifiability_scores.append(verifiability_score)
        
        # 计算平均值
        avg_unambiguity = np.mean(unambiguity_scores) if unambiguity_scores else 0.0
        avg_verifiability = np.mean(verifiability_scores) if verifiability_scores else 0.0
        
        return {
            "consistency": round(consistency, 3),
            "completeness": round(completeness, 3),
            "unambiguity": round(avg_unambiguity, 3),
            "verifiability": round(avg_verifiability, 3),
            "matched_count": matched_count,
            "total_golden": total_golden,
            "total_generated": len(generated_reqs),
            "match_rate": round(matched_count / total_golden * 100, 1) if total_golden > 0 else 0.0
        }


def load_golden_requirements(filepath: str) -> List[Dict]:
    """加载黄金需求集"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}")
        sys.exit(1)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON文件结构: {type(data)}")
        print(f"JSON文件中的键: {list(data.keys())}")
        
        # 从requirement_modules中提取所有需求
        if "requirement_modules" in data and isinstance(data["requirement_modules"], list):
            print(f"找到requirement_modules，包含 {len(data['requirement_modules'])} 个模块")
            
            all_requirements = []
            
            for module in data["requirement_modules"]:
                if isinstance(module, dict) and "requirements" in module:
                    module_name = module.get("module_name", "未知模块")
                    requirements = module["requirements"]
                    
                    if isinstance(requirements, list):
                        print(f"  从模块 '{module_name}' 提取到 {len(requirements)} 条需求")
                        all_requirements.extend(requirements)
            
            if all_requirements:
                print(f"总共提取到 {len(all_requirements)} 条黄金需求")
                return all_requirements
        
        print(f"警告: 无法解析黄金需求文件格式")
        return []
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"读取文件错误: {e}")
        return []


def load_generated_requirements(filepath: str) -> List[Tuple[str, str]]:
    """加载生成的需求文件"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"错误: 文件不存在 {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按行解析需求
    requirements = []
    lines = content.strip().split('\n')
    
    for line in lines:
        if line.strip():
            req_type, req_content = extract_requirement(line)
            if req_content:  # 只要有内容就加入
                requirements.append((req_type, req_content))
    
    return requirements


def main():
    """主函数"""
    print("=" * 60)
    print("优化的需求评估系统 - 一致性、完全性、明确性、可验证性评估")
    print("=" * 60)
    
    # 配置文件路径
    MODEL_PATH = "D:\\study\\ai-tools\\work\\models"
    GOLDEN_PATH = "d:\\study\\ai-tools\\work\\requirement\\requirement.json"
    
    # 框架结果文件
    FRAMEWORK_FILES = {
        "AIRS": "d:\\study\\ai-tools\\work\\horizontal-experiment\\Qwen3-Max\\result\\airs_result.txt",
        "COT": "d:\\study\\ai-tools\\work\\horizontal-experiment\\Qwen3-Max\\result\\cot_result.txt", 
        "REACT": "d:\\study\\ai-tools\\work\\horizontal-experiment\\Qwen3-Max\\result\\react_result.txt",
        "SELF-ASK": "d:\\study\\ai-tools\\work\\horizontal-experiment\\Qwen3-Max\\result\\selfask_result.txt"
    }
    
    # 加载黄金需求集
    print(f"加载黄金需求集: {GOLDEN_PATH}")
    golden_requirements = load_golden_requirements(GOLDEN_PATH)
    print(f"黄金需求数量: {len(golden_requirements)}")
    
    if len(golden_requirements) == 0:
        print("错误: 黄金需求集为空，无法进行评估")
        return
    
    # 初始化评估器
    evaluator = OptimizedRequirementEvaluator(MODEL_PATH)
    
    # 评估每个框架
    results = {}
    for name, filepath in FRAMEWORK_FILES.items():
        print(f"\n评估框架: {name}")
        
        # 加载生成的需求
        generated_requirements = load_generated_requirements(filepath)
        print(f"  生成需求数量: {len(generated_requirements)}")
        
        if len(generated_requirements) == 0:
            print(f"  警告: {name} 生成的需求为空")
            results[name] = {
                "consistency": 0.0,
                "completeness": 0.0,
                "unambiguity": 0.0,
                "verifiability": 0.0,
                "matched_count": 0,
                "total_golden": len(golden_requirements),
                "total_generated": 0,
                "match_rate": 0.0
            }
            continue
        
        # 评估所有指标
        evaluation_result = evaluator.evaluate_all_metrics(golden_requirements, generated_requirements)
        results[name] = evaluation_result
        
        # 打印结果
        print(f"  匹配需求数: {evaluation_result['matched_count']}/{evaluation_result['total_golden']} ({evaluation_result['match_rate']}%)")
        print(f"  一致性: {evaluation_result['consistency']:.3f}")
        print(f"  完全性: {evaluation_result['completeness']:.3f}")
        print(f"  明确性: {evaluation_result['unambiguity']:.3f}")
        print(f"  可验证性: {evaluation_result['verifiability']:.3f}")
    
    # 保存结果到文件
    output_file = Path(__file__).parent / "result.txt"
    print(f"\n保存结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("框架对比评估结果\n")
        f.write("=" * 80 + "\n")
        f.write("评估标准说明:\n")
        f.write("1. 一致性(0-1): 生成需求与黄金需求的语义相似度平均值\n")
        f.write("2. 完全性(0-1): 匹配的黄金需求比例（相似度≥0.7）\n")
        f.write("3. 明确性(0-1): 基于模糊词比例，1表示完全没有模糊词\n")
        f.write("4. 可验证性(0-1): 基于可验证特征数，使用S型函数映射到0-1\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'框架':<12} {'一致性':<10} {'完全性':<10} {'明确性':<10} {'可验证性':<10} {'匹配需求':<15}\n")
        f.write("-" * 80 + "\n")
        
        for name in ["AIRS", "COT", "REACT", "SELF-ASK"]:
            if name in results:
                r = results[name]
                f.write(f"{name:<12} {r['consistency']:<10.3f} {r['completeness']:<10.3f} "
                       f"{r['unambiguity']:<10.3f} {r['verifiability']:<10.3f} "
                       f"{r['matched_count']}/{r['total_golden']} ({r['match_rate']}%)\n")
        
        # 添加详细结果
        f.write("\n" + "=" * 80 + "\n")
        f.write("详细结果:\n")
        
        for name in ["AIRS", "COT", "REACT", "SELF-ASK "]:
            if name in results:
                r = results[name]
                f.write(f"\n{name}:\n")
                f.write(f"  生成需求数量: {r['total_generated']}\n")
                f.write(f"  一致性: {r['consistency']:.3f}\n")
                f.write(f"  完全性: {r['completeness']:.3f}\n")
                f.write(f"  明确性: {r['unambiguity']:.3f}\n")
                f.write(f"  可验证性: {r['verifiability']:.3f}\n")
                f.write(f"  匹配需求: {r['matched_count']}/{r['total_golden']} ({r['match_rate']}%)\n")
    
    print("\n评估完成!")


if __name__ == "__main__":
    # 初始化jieba分词器
    jieba.initialize()
    main()
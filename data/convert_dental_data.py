import json
from tqdm import tqdm

def convert_qa_data(input_path, output_path):
    """
    转换牙科问答数据为EasyEdit兼容格式
    """
    with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(f, desc="转换问答数据"):
            data = json.loads(line.strip())
            # EasyEdit问答数据格式要求
            easyedit_data = {
                "instruction": "请作为一名专业的牙科医生，回答用户的口腔相关问题。",
                "input": data["conversations"][0]["content"],
                "output": data["conversations"][1]["content"]
            }
            out_f.write(json.dumps(easyedit_data, ensure_ascii=False) + '\n')

def convert_choice_data(input_path, output_path):
    """
    转换牙科选择题数据为EasyEdit兼容格式（强化选择题推理逻辑）
    输入格式：Options字段包含A-E选项，以换行分隔
    输出格式：EasyEdit的instruction/input/output格式
    """
    # 统计处理情况，方便排查问题
    processed_count = 0
    error_count = 0
    print(f"{processed_count}")
    with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(f, desc="转换选择题数据"):
            try:
                # 加载单行JSON数据
                data = json.loads(line.strip())
                
                # 核心修复：解析Options字段中的A/B/C/D/E选项
                options_str = data.get('Options', '')
                # 按换行分割选项，并过滤空行
                option_lines = [line.strip() for line in options_str.split('\n') if line.strip()]
                # 构建选项字典（键：A/B/C/D/E，值：选项内容）
                option_dict = {}
                for opt_line in option_lines:
                    # 处理 "A 片状白斑" 这种格式（兼容全角/半角空格）
                    if len(opt_line) >= 2 and opt_line[1] in [' ', '　']:
                        opt_key = opt_line[0]  # 提取A/B/C/D/E
                        opt_value = opt_line[2:].strip()  # 提取选项内容
                        option_dict[opt_key] = opt_value
                
                # 构造标准化prompt（使用解析后的选项）
                input_content = (
                    f"请回答以下牙科选择题，先给出答案选项，再简要说明理由。"
                    f"题干：{data['Question']} "
                    f"选项：A.{option_dict.get('A', '')} B.{option_dict.get('B', '')} "
                    f"C.{option_dict.get('C', '')} D.{option_dict.get('D', '')} E.{option_dict.get('E', '')}"
                )
                
                # 构造输出内容
                output_content = f"答案：{data['Answer']} 解析：{data['Explanation']}"
                
                # EasyEdit格式封装
                easyedit_data = {
                    "instruction": "请作为一名专业的牙科医生，解答口腔医学相关选择题。",
                    "input": input_content,
                    "output": output_content
                }
                
                # 写入输出文件（保证JSON格式正确，禁用ASCII转义）
                out_f.write(json.dumps(easyedit_data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except Exception as e:
                # 捕获异常，避免程序中断，方便排查错误数据
                error_count += 1
                print(f"\n处理行时出错：{e}，行内容：{line[:100]}")
    
    # 输出处理统计
    print(f"\n转换完成！成功处理：{processed_count} 条，失败：{error_count} 条")

if __name__ == "__main__":
    # 替换为你的数据路径
    QA_INPUT_PATH = "./huatuo_dental_qa.jsonl"
    QA_OUTPUT_PATH = "./easyedit_dental_qa.jsonl"
    CHOICE_INPUT_PATH = "./cmexam_dental_choice.jsonl"
    CHOICE_OUTPUT_PATH = "./easyedit_dental_choice.jsonl"
    
    # 转换数据
    convert_qa_data(QA_INPUT_PATH, QA_OUTPUT_PATH)
    # convert_choice_data(CHOICE_INPUT_PATH, CHOICE_OUTPUT_PATH)
    
    print("数据转换完成！生成的EasyEdit兼容数据已保存至当前目录。")

"""
Description: 将训练数据转换成DeepSeek R1接受的数据格式
             格式示例：
             [
  {
    "conversations": [
      {
        "from": "user",
        "value": "总结下面这段文本的关键词，随着互联网的普及，电子商务也发展得飞快。越来越多的人开始在网上购物，这让网络商品生意越来越好，也不断地涌现了新的电商平台。电商不仅改变了人们的消费习惯，也让零售业变得更加全球化，商品范围更加广泛。在这一过程中，电商物流成为了关键的一环。电商物流涉及到了商品配送、订单跟踪、包装和送货等环节。一个高效的物流系统可以让消费者更快地收到商品，也可以让卖家更好地控制自己的货物。因此，电商物流成为了电商行业中的重要一环。目前，国内的电商物流企业包括顺丰、圆通、申通、韵达、中通等大型企业。这些企业也在不断提升服务水平，以应对日益增长的市场需求。未来，电商物流的市场规模仍有较大空间，希望更多的企业加入进来，为电商行业的繁荣发展做出更大的贡献。",
      },
      {
        "from": "assistant",
        "value": "互联网，电子商务，电商平台，消费习惯，电商物流，配送，订单跟踪，包装，送货，顺丰，圆通，申通，韵达，中通，服务水平，市场规模，企业。"
      }
    ],
    "id": "identity_2"
  }
]
    
-*- Encoding: UTF-8 -*-
@File     ：organise_data.py
@Author   ：King Songtao
@Time     ：2025/2/26 下午7:42
@Contact  ：king.songtao@gmail.com
"""
import json
import re


def process_json_file(input_file_path, output_file_path):
    """
    读取JSON文件并转换为指定格式

    参数:
    input_file_path (str): 输入JSON文件路径
    output_file_path (str): 输出JSON文件路径
    """
    # 读取输入文件
    input_data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                input_data.append(json.loads(line))

    # 转换数据
    output_data = []
    for i, item in enumerate(input_data):
        context = item.get('context', '')
        target = item.get('target', '')

        # 提取instruction内容
        instruction_match = re.search(r'Instruction: (.*?)(?=\nAnswer: )', context, re.DOTALL)
        instruction = instruction_match.group(1) if instruction_match else ""

        # 清理target内容，移除```json\n和\n```
        cleaned_target = re.sub(r'```json\n', '', target)
        cleaned_target = re.sub(r'\n```', '', cleaned_target)

        # 创建新的数据结构
        converted_item = {
            "conversations": [
                {
                    "from": "user",
                    "value": instruction
                },
                {
                    "from": "assistant",
                    "value": cleaned_target
                }
            ],
            "id": f"identity_{i}"
        }

        output_data.append(converted_item)

    # 写入输出文件，单行格式
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)

    print(f"转换完成! 共处理了 {len(output_data)} 条数据")
    print(f"输出文件保存在: {output_file_path}")


if __name__ == "__main__":
    input_file = "/root/autodl-tmp/data/sop_cla_datasets/mixed_dev_dataset.jsonl"  # 替换为你的输入文件路径
    output_file = "/root/autodl-tmp/data/datasets/dev.json"  # 替换为你希望的输出文件路径

    process_json_file(input_file, output_file)

import os
import json
import torch
import ast
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model(model_path):
    """加载Qwen2.5-VL模型和处理器"""
    print("正在加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    print("模型加载完成")
    return model, processor

def detect_objects(model, processor, image_path):
    """使用Qwen2.5-VL模型检测图像中的对象"""
    # 打开图像
    image = Image.open(image_path)

    # 定义系统提示和用户提示，优化目标检测效果
    system_prompt = '''You are an expert in object detection, specifically for identifying cardboard boxes (cartons) in images. Your task is to detect and provide precise bounding boxes for ALL cardboard boxes visible in the image.

Your response must include:
1. Accurate bounding box coordinates in the format [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
2. The label "carton" for each box

Critical detection rules:
- ONLY detect cardboard boxes (carton), ignore all other objects
- Detect each individual box separately, especially when multiple boxes are present
- Provide precise coordinates that tightly bound each box
- Avoid grouping multiple boxes into a single large bounding box
- If you see multiple small boxes, detect each one individually rather than combining them
- ONLY detect boxes you are extremely confident about (confidence > 90%)
- DO NOT detect boxes that are heavily occluded, blurry, or unclear
- DO NOT detect reflections, shadows, or background patterns as boxes
- AVOID duplicate detections of the same box
- If in doubt, do not include the box in the results
- Count and detect all visible boxes, even if they are at different distances or angles
- Pay special attention to boxes in the background or at the edges of the image
- Boxes may be of different sizes, orientations, and colors but must be cardboard boxes

Return the results in strict JSON format:
[{"bbox_2d": [x1, y1, x2, y2], "label": "carton"}, ...]

Example format:
[{"bbox_2d": [100, 150, 200, 250], "label": "carton"}]'''
    user_prompt = '''Carefully analyze the image and detect ALL cardboard boxes (carton). For each box, provide:
1. Accurate bounding box coordinates in the format [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
2. The label "carton" for each box

Critical detection rules:
- ONLY detect cardboard boxes (carton), ignore all other objects
- Detect each individual box separately, especially when multiple boxes are present
- Provide precise coordinates that tightly bound each box
- Avoid grouping multiple boxes into a single large bounding box
- If you see multiple small boxes, detect each one individually rather than combining them
- ONLY detect boxes you are extremely confident about (confidence > 90%)
- DO NOT detect boxes that are heavily occluded, blurry, or unclear
- DO NOT detect reflections, shadows, or background patterns as boxes
- AVOID duplicate detections of the same box
- If in doubt, do not include the box in the results
- Count and detect all visible boxes, even if they are at different distances or angles
- Pay special attention to boxes in the background or at the edges of the image
- Boxes may be of different sizes, orientations, and colors but must be cardboard boxes

Return the results in strict JSON format:
[{"bbox_2d": [x1, y1, x2, y2], "label": "carton"}, ...]

Example format:
[{"bbox_2d": [100, 150, 200, 250], "label": "carton"}]'''

    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": image_path}
        ]}
    ]

    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 处理输入
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    # 生成输出，调整参数以提高准确性
    generation_config = {
        "max_new_tokens": 2048,  # 适当减少最大令牌数以提高响应速度
        "do_sample": False,  # 使用贪婪解码以提高一致性
        "temperature": None,  # 明确设置为None以避免警告
        "top_p": None,  # 明确设置为None以避免警告
        "repetition_penalty": 1.2,  # 增加重复惩罚以避免重复输出
        "length_penalty": 1.0,  # 保持长度惩罚以鼓励完整输出
        "num_beams": 3  # 使用束搜索以提高输出质量
    }

    # 过滤掉None值
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    output_ids = model.generate(**inputs, **generation_config)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return output_text[0]

def parse_detection_result(result_text, image_size=None):
    """解析检测结果，提取边界框信息"""
    try:
        # 查找JSON部分
        start_idx = result_text.find('```json')
        if start_idx != -1:
            start_idx += 7  # 跳过```json
            end_idx = result_text.find('```', start_idx)
            if end_idx == -1:
                end_idx = len(result_text)
            json_text = result_text[start_idx:end_idx].strip()
        else:
            # 查找第一个左方括号和最后一个右方括号
            start_idx = result_text.find('[')
            end_idx = result_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_text = result_text[start_idx:end_idx]
            else:
                json_text = result_text

        # 清理JSON文本，确保格式正确
        json_text = json_text.strip()
        if json_text.startswith('```') and json_text.endswith('```'):
            json_text = json_text[3:-3].strip()
        if json_text.startswith('json'):
            json_text = json_text[4:].strip()

        # 尝试修复常见的JSON格式问题
        # 确保标签字段使用双引号
        import re
        json_text = re.sub(r"'label':\s*'([^']*)'", r'"label": "\1"', json_text)
        json_text = re.sub(r"'bbox_2d':", '"bbox_2d":', json_text)
        json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)  # 将单引号键替换为双引号

        # 解析JSON
        detection_result = json.loads(json_text)

        # 后处理：确保每个检测结果都有正确的格式
        processed_result = []
        for obj in detection_result:
            if isinstance(obj, dict) and 'bbox_2d' in obj and 'label' in obj:
                # 确保标签是字符串
                label = str(obj['label']).lower().strip()
                # 标准化标签
                if 'box' in label and 'carton' not in label:
                    label = 'carton'
                elif label in ['boxes', 'box']:
                    label = 'carton'

                # 确保边界框坐标是数字
                bbox = obj['bbox_2d']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # 确保坐标是整数
                    bbox = [int(coord) for coord in bbox]
                    # 确保坐标顺序正确 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = bbox
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    bbox = [x1, y1, x2, y2]

                    # 验证检测框是否在图像范围内
                    if image_size:
                        img_width, img_height = image_size
                        x1 = max(0, min(x1, img_width))
                        x2 = max(0, min(x2, img_width))
                        y1 = max(0, min(y1, img_height))
                        y2 = max(0, min(y2, img_height))
                        # 确保检测框有最小尺寸
                        if x2 - x1 > 5 and y2 - y1 > 5:
                            bbox = [x1, y1, x2, y2]
                            processed_result.append({
                                'bbox_2d': bbox,
                                'label': label
                            })
                    else:
                        processed_result.append({
                            'bbox_2d': bbox,
                            'label': label
                        })

        # 进一步优化：去除重复或重叠的检测框
        processed_result = remove_duplicate_detections(processed_result, iou_threshold=0.3)  # 调整IoU阈值以平衡去重和完整性

        return processed_result
    except Exception as e:
        print(f"解析检测结果时出错: {e}")
        print(f"原始结果: {result_text}")
        # 尝试使用更强大的解析方法
        try:
            # 清理文本
            cleaned_text = result_text.replace('```json', '').replace('```', '').strip()
            # 尝试修复常见的格式问题
            cleaned_text = cleaned_text.strip()

            # 使用正则表达式提取JSON数组
            import re
            json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                # 尝试修复更多的格式问题
                json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)  # 将单引号键替换为双引号
                json_text = re.sub(r"'label':\s*'([^']*)'", r'"label": "\1"', json_text)
                json_text = re.sub(r"'bbox_2d':", '"bbox_2d":', json_text)
                json_text = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_text)  # 为未加引号的键添加引号
                json_text = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)', r': "\1"', json_text)  # 为未加引号的字符串值添加引号

                # 解析JSON
                detection_result = json.loads(json_text)

                # 后处理
                processed_result = []
                for obj in detection_result:
                    if isinstance(obj, dict) and 'bbox_2d' in obj and 'label' in obj:
                        label = str(obj['label']).lower().strip()
                        if 'box' in label and 'carton' not in label:
                            label = 'carton'
                        elif label in ['boxes', 'box']:
                            label = 'carton'

                        bbox = obj['bbox_2d']
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            bbox = [int(coord) for coord in bbox]
                            x1, y1, x2, y2 = bbox
                            if x1 > x2:
                                x1, x2 = x2, x1
                            if y1 > y2:
                                y1, y2 = y2, y1
                            bbox = [x1, y1, x2, y2]

                            # 验证检测框是否在图像范围内
                            if image_size:
                                img_width, img_height = image_size
                                x1 = max(0, min(x1, img_width))
                                x2 = max(0, min(x2, img_width))
                                y1 = max(0, min(y1, img_height))
                                y2 = max(0, min(y2, img_height))
                                # 确保检测框有最小尺寸
                                if x2 - x1 > 5 and y2 - y1 > 5:
                                    bbox = [x1, y1, x2, y2]
                                    processed_result.append({
                                        'bbox_2d': bbox,
                                        'label': label
                                    })
                            else:
                                processed_result.append({
                                    'bbox_2d': bbox,
                                    'label': label
                                })

                # 进一步优化：去除重复或重叠的检测框
                processed_result = remove_duplicate_detections(processed_result, iou_threshold=0.3)

                return processed_result
        except Exception as e2:
            print(f"备选方案解析也失败: {e2}")
            pass

        return []

def remove_duplicate_detections(detections, iou_threshold=0.5):
    """去除重复或重叠的检测框"""
    if len(detections) <= 1:
        return detections

    # 按置信度排序（这里我们假设所有检测框的置信度相同，按面积排序）
    # 计算每个检测框的面积
    detections_with_area = []
    for det in detections:
        bbox = det['bbox_2d']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        detections_with_area.append((det, area))

    # 按面积排序（从大到小），优先保留大的检测框
    detections_with_area.sort(key=lambda x: x[1], reverse=True)

    # 使用非极大值抑制（NMS）去除重复检测
    suppressed = set()

    for i, (det_i, area_i) in enumerate(detections_with_area):
        if i in suppressed:
            continue

        # 检查后续的检测框是否与当前检测框重叠
        for j, (det_j, area_j) in enumerate(detections_with_area[i+1:], i+1):
            if j in suppressed:
                continue

            # 计算IoU
            bbox_i = det_i['bbox_2d']
            bbox_j = det_j['bbox_2d']

            # 计算交集
            x1_inter = max(bbox_i[0], bbox_j[0])
            y1_inter = max(bbox_i[1], bbox_j[1])
            x2_inter = min(bbox_i[2], bbox_j[2])
            y2_inter = min(bbox_i[3], bbox_j[3])

            # 检查是否有交集
            if x1_inter < x2_inter and y1_inter < y2_inter:
                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

                # 计算并集
                area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                union_area = area_i + area_j - inter_area

                # 计算IoU
                if union_area > 0:
                    iou = inter_area / union_area

                    # 如果IoU超过阈值，则抑制较小的检测框
                    if iou > iou_threshold:
                        suppressed.add(j)

    # 过滤掉被抑制的检测框
    final_keep = []
    for i, (det, _) in enumerate(detections_with_area):
        if i not in suppressed:
            final_keep.append(det)

    return final_keep

def draw_bounding_boxes(image_path, detection_result, output_path):
    """在图像上绘制边界框并保存"""
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 尝试使用适中的字体大小以提高可见性
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 20)  # 减小字体到20
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)  # 减小字体到20
        except:
            font = ImageFont.load_default()

    # 为不同标签定义颜色
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'lime']

    # 绘制每个检测到的对象
    for i, obj in enumerate(detection_result):
        if 'bbox_2d' in obj and len(obj['bbox_2d']) == 4 and 'label' in obj:
            bbox = obj['bbox_2d']
            label = obj['label']

            # 确保边界框坐标正确
            x1, y1, x2, y2 = bbox
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # 选择颜色
            color = colors[i % len(colors)]

            # 绘制边界框（增加宽度到4以提高可见性）
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            # 绘制标签背景以增强可见性
            # 计算文本大小
            try:
                # 对于较新版本的PIL
                text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)  # 将标签位置移到检测框外侧上方
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # 对于较旧版本的PIL
                text_width, text_height = draw.textsize(label, font=font)

            # 绘制标签背景框（稍微扩大背景框以提高可见性，位置在检测框外侧上方）
            draw.rectangle([x1-2, y1 - text_height - 2, x1 + text_width+2, y1], fill=color)

            # 绘制标签文本（使用白色字体以增强对比度，并添加轻微阴影效果，位置在检测框外侧上方）
            draw.text((x1+1, y1 - text_height + 1), label, fill="black", font=font)  # 添加黑色阴影
            draw.text((x1, y1 - text_height), label, fill="white", font=font)  # 主要白色文本

    # 保存图像
    image.save(output_path)
    print(f"结果图像已保存到: {output_path}")

def save_detection_result(detection_result, output_path):
    """保存检测结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detection_result, f, ensure_ascii=False, indent=2)
    print(f"检测结果已保存到: {output_path}")


def save_yolo_labels(detection_result, image_size, output_path):
    """保存YOLO格式的标签文件"""
    img_width, img_height = image_size
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in detection_result:
            if 'bbox_2d' in obj and len(obj['bbox_2d']) == 4 and 'label' in obj:
                # 获取边界框坐标
                x1, y1, x2, y2 = obj['bbox_2d']
                
                # 转换为YOLO格式: class_id center_x center_y width height (normalized)
                # 假设"carton"类的class_id为0
                class_id = 0
                
                # 计算中心点和宽高
                center_x = (x1 + x2) / 2.0 / img_width
                center_y = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # 写入YOLO格式标签
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"YOLO标签已保存到: {output_path}")

def main():
    # 设置路径
    model_path = "D:/无锡捷普迅智能科技有限公司/Qwen2.5-VL/models"
    images_folder = "D:/无锡捷普迅智能科技有限公司/Qwen2.5-VL/datasets/Data/test"
    outputs_folder = "D:/无锡捷普迅智能科技有限公司/Qwen2.5-VL/outputs"
    
    # 创建输出子目录
    images_output_folder = os.path.join(outputs_folder, "images")
    labels_output_folder = os.path.join(outputs_folder, "labels")
    json_output_folder = os.path.join(outputs_folder, "json")
    raw_output_folder = os.path.join(outputs_folder, "raw")
    
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(labels_output_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)
    os.makedirs(raw_output_folder, exist_ok=True)
    
    # 加载模型
    model, processor = load_model(model_path)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for i, image_file in enumerate(image_files):
        print(f"\n处理图像 ({i+1}/{len(image_files)}): {image_file}")
        
        # 构建文件路径
        image_path = os.path.join(images_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        try:
            # 打开图像以获取尺寸信息
            image = Image.open(image_path)
            image_size = image.size  # (width, height)
            
            # 检测对象
            result_text = detect_objects(model, processor, image_path)
            print(f"检测结果: {result_text}")
            
            # 保存原始结果
            raw_output_path = os.path.join(raw_output_folder, f"{base_name}_raw.txt")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            print(f"原始输出已保存到: {raw_output_path}")
            
            # 解析检测结果，传递图像尺寸信息
            detection_result = parse_detection_result(result_text, image_size)
            print(f"解析后的检测结果: {detection_result}")
            print(f"检测到 {len(detection_result)} 个对象")
            
            if detection_result:
                # 保存JSON结果
                json_output_path = os.path.join(json_output_folder, f"{base_name}_detection.json")
                save_detection_result(detection_result, json_output_path)
                
                # 绘制边界框并保存图像
                image_output_path = os.path.join(images_output_folder, f"{base_name}_detection.jpg")
                draw_bounding_boxes(image_path, detection_result, image_output_path)
                
                # 保存YOLO格式标签文件
                labels_output_path = os.path.join(labels_output_folder, f"{base_name}.txt")
                save_yolo_labels(detection_result, image_size, labels_output_path)
                
                print(f"成功处理图像: {image_file}")
            else:
                print(f"未检测到对象或解析失败: {image_file}")
                
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {e}")
            continue

if __name__ == "__main__":
    main()
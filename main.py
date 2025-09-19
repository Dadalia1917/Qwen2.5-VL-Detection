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
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("模型加载完成")
    return model, processor

def detect_objects(model, processor, image_path):
    """使用Qwen2.5-VL模型检测图像中的对象"""
    # 打开图像
    image = Image.open(image_path)

    # 定义系统提示和用户提示
    system_prompt = "You are a helpful assistant specializing in object detection. Please identify and locate objects accurately with their bounding box coordinates."
    user_prompt = "Please detect all cardboard boxes in this image. For each box, provide precise bounding box coordinates in JSON format: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'carton'}]. Only detect clearly visible cardboard boxes."

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

    # 生成输出
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return output_text[0]

def parse_detection_result(result_text, image_size=None):
    """解析检测结果，提取边界框信息"""
    try:
        import re
        
        # 清理文本，提取JSON部分
        cleaned_text = result_text
        if '```json' in cleaned_text:
            cleaned_text = cleaned_text.split('```json')[1].split('```')[0]
        elif '[' in cleaned_text and ']' in cleaned_text:
            # 提取JSON数组部分
            start = cleaned_text.find('[')
            end = cleaned_text.rfind(']') + 1
            cleaned_text = cleaned_text[start:end]
        
        # 修复常见的JSON格式问题
        cleaned_text = re.sub(r"'([^']*)':", r'"\1":', cleaned_text)
        cleaned_text = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned_text)
        
        # 解析JSON
        detection_result = json.loads(cleaned_text)
        
        processed_result = []
        for obj in detection_result:
            if isinstance(obj, dict) and 'bbox_2d' in obj and 'label' in obj:
                bbox = [int(coord) for coord in obj['bbox_2d']]
                x1, y1, x2, y2 = bbox
                
                # 确保坐标顺序正确
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # 验证检测框有效性
                if image_size:
                    img_width, img_height = image_size
                    x1 = max(0, min(x1, img_width))
                    x2 = max(0, min(x2, img_width))
                    y1 = max(0, min(y1, img_height))
                    y2 = max(0, min(y2, img_height))
                    
                    if x2 - x1 > 5 and y2 - y1 > 5:
                        processed_result.append({
                            'bbox_2d': [x1, y1, x2, y2],
                            'label': 'carton'
                        })
                else:
                    processed_result.append({
                        'bbox_2d': [x1, y1, x2, y2],
                        'label': 'carton'
                    })
        
        return remove_duplicate_detections(processed_result)
        
    except Exception as e:
        print(f"解析检测结果时出错: {e}")
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
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    for i, obj in enumerate(detection_result):
        if 'bbox_2d' in obj and 'label' in obj:
            x1, y1, x2, y2 = obj['bbox_2d']
            label = obj['label']
            color = colors[i % len(colors)]
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            draw.text((x1, y1 - 20), label, fill=color, font=font)
    
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
        
        image_path = os.path.join(images_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        try:
            image = Image.open(image_path)
            image_size = image.size
            
            # 检测对象
            result_text = detect_objects(model, processor, image_path)
            
            # 保存原始结果
            raw_output_path = os.path.join(raw_output_folder, f"{base_name}_raw.txt")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            # 解析检测结果
            detection_result = parse_detection_result(result_text, image_size)
            print(f"检测到 {len(detection_result)} 个对象")
            
            if detection_result:
                # 保存JSON结果
                json_output_path = os.path.join(json_output_folder, f"{base_name}_detection.json")
                save_detection_result(detection_result, json_output_path)
                
                # 绘制边界框
                image_output_path = os.path.join(images_output_folder, f"{base_name}_detection.jpg")
                draw_bounding_boxes(image_path, detection_result, image_output_path)
                
                # 保存YOLO格式标签
                labels_output_path = os.path.join(labels_output_folder, f"{base_name}.txt")
                save_yolo_labels(detection_result, image_size, labels_output_path)
                
                print(f"完成处理: {image_file}")
            else:
                print(f"未检测到对象: {image_file}")
                
        except Exception as e:
            print(f"处理错误 {image_file}: {e}")
            continue

if __name__ == "__main__":
    main()
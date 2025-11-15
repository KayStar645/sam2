import json

# Đọc notebook
with open('ViT/ViT.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Tìm cell chứa sam_vit_results
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'sam_vit_results = {' in source:
            print(f"Found at cell {i}")
            # Tạo code mới
            new_code = """# Kết quả từ SAM-ViT (bổ sung đầy đủ metrics để so sánh tương đương với SwinT-YOLOv8-p2)
# Tính Mean IoU và Dice từ segmentation_results (nếu có)
mean_iou_list = []
mean_dice_list = []
if 'segmentation_results' in globals() and segmentation_results:
    for split_results in segmentation_results.values():
        for class_stats in split_results.get('class_stats', {}).values():
            if 'mean_iou' in class_stats:
                mean_iou_list.append(class_stats['mean_iou'])
            if 'mean_dice' in class_stats:
                mean_dice_list.append(class_stats['mean_dice'])

mean_iou = np.mean(mean_iou_list) if mean_iou_list else 0.0
mean_dice = np.mean(mean_dice_list) if mean_dice_list else 0.0

# Tính mean density từ density_results (nếu có)
mean_density_list = []
if 'density_results' in globals() and density_results:
    for dens_results in density_results.values():
        if 'mean_instances_per_image' in dens_results:
            mean_density_list.append(dens_results['mean_instances_per_image'])

mean_density = np.mean(mean_density_list) if mean_density_list else 0.0

sam_vit_results = {
    'Accuracy': test_results['accuracy'] / 100,  # Chuyển từ % sang decimal (tương đương mAP@0.5)
    'F1_score': test_results['f1'],
    'Precision': test_results['precision'],
    'Recall': test_results['recall'],
    # Segmentation metrics từ SAM (ưu điểm của SAM-ViT so với YOLO)
    'Mean_IoU': mean_iou,      # IoU trung bình từ SAM masks
    'Mean_Dice': mean_dice,    # Dice coefficient trung bình từ SAM masks
    # Performance metrics
    'FPS': 'N/A (2-stage pipeline)',  # Chậm hơn do 2-stage: SAM + ViT
    # Density analysis (ưu điểm của SAM-ViT - xác định mật độ chính xác)
    'Mean_density_per_image': mean_density,  # Số cá thể rầy trung bình/ảnh
    # Per-class performance details
    'Per_class_metrics': test_results.get('per_class', {}),
    'description': 'SAM-ViT: Segmentation (SAM) + Classification (ViT) - 2-stage pipeline cho phân đoạn chi tiết và xác định mật độ'
}
"""
            # Tìm phần cần thay thế trong source
            lines = source.split('\n')
            new_lines = []
            skip_until_brace = False
            in_sam_vit = False
            brace_count = 0
            
            for j, line in enumerate(lines):
                if 'sam_vit_results = {' in line:
                    in_sam_vit = True
                    brace_count = line.count('{') - line.count('}')
                    # Thêm code mới
                    new_lines.append(new_code)
                    skip_until_brace = True
                    continue
                
                if skip_until_brace:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0 and '}' in line:
                        skip_until_brace = False
                        # Tìm dòng tiếp theo sau }
                        continue
                    else:
                        continue
                
                new_lines.append(line)
            
            # Cập nhật source
            cell['source'] = new_lines
            break

# Lưu lại notebook
with open('ViT/ViT.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")


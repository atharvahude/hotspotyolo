from hotspotyolo import yolo_heatmap

image_path = "/home/atharvahude/Desktop/yolov8-explainer/obb-blackbox-attacks/dataset/patch-attack-images/100000003.bmp"
model_weight = '/home/atharvahude/Desktop/yolov8-explainer/YOLOv8_Explainer_old/images/Ships-Experiment-5-Folds-Models-Dataset/models/train4/weights/best.pt'
output_folder = 'test_result'


params = {
    'weight': model_weight,
    'device': 'cuda:0',
    'method': 'GradCAMPlusPlus', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
    'layer': [21],
    'backward_type': 'all', # detect:<class, box, all> segment:<class, box, segment, all> pose:<box, keypoint, all> obb:<box, angle, all> classify:<all>
    'conf_threshold': 0.2, # 0.2
    'ratio': 0.02, # 0.02-0.1
    'show_result': True, # Set to False if you do not need to draw results
    'renormalize': True, # Set to True to restrict the heatmap within the bounding box (only effective for detect, segment, pose)
    'task':'obb', # Task (detect, segment, pose, obb, classify)
    'img_size':1280, # Image size
    'save_metadata': True, # Save metadata in the output folder
}

model = yolo_heatmap(**params)
model(image_path, output_folder)


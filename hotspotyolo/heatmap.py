import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil, copy
import numpy as np
np.random.seed(0)
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import  non_max_suppression
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import json
from .activations_and_gradients import ActivationsAndGradients
from .targets import *
from .utils import letterbox
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM


class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize, task, img_size,save_metadata=True):
        self.save_metadata = save_metadata
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False
        
        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")
        
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2) # Draw bounding box.
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)  # Draw category and confidence.
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path, save_path):
        # img process
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except:
            print(f"Warning... {img_path} read failure.")
            return
        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True) # If you want to completely fix it to have equal width and height, set auto to False.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        print(f'tensor size:{tensor.size()}')
        
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure.")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        pred = self.model_yolo.predict(tensor, conf=self.conf_threshold)[0]
        if self.renormalize and self.task in ['detect', 'segment', 'pose']:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
        if self.show_result:
            cam_image = pred.plot(img=cam_image,
                                  conf=True, # Display confidence
                                  font_size=None, # Font size, None means calculated based on the current image size
                                  line_width=None, # Line width, None means calculated based on the current image size
                                  labels=True, # Display labels
                                  )
        
        # Remove padding borders
        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
    
    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')

        if self.save_metadata:
            metadata = {
                'weight': self.weight,
                'device': str(self.device),
                'method': self.method.__class__.__name__,
                'layer': [l.__class__.__name__ for l in self.target_layers],
                'layer_indices': self.layer,
                'backward_type': self.backward_type,
                'conf_threshold': self.conf_threshold,
                'ratio': self.ratio,
                'show_result': self.show_result,
                'renormalize': self.renormalize,
                'task': self.task,
                'img_size': self.img_size
            }
            with open(f'{save_path}/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {save_path}")

    
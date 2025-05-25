import torch
from tqdm import trange

class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result)

class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result)

class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result)

class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)
    
    def forward(self, data):
        return data.max()
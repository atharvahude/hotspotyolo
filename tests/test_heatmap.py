import unittest
from hotspotyolo.heatmap import yolo_heatmap
import os
class TestYoloHeatmap(unittest.TestCase):

    def setUp(self):
        self.params = {
            'weight': 'path/to/weights.pt',
            'device': 'cuda:0',
            'method': 'GradCAMPlusPlus',
            'layer': [21],
            'backward_type': 'all',
            'conf_threshold': 0.2,
            'ratio': 0.02,
            'show_result': True,
            'renormalize': False,
            'task': 'detect',
            'img_size': 640,
        }
        self.model = yolo_heatmap(**self.params)

    def test_process_single_image(self):
        result_path = 'test_result.png'
        img_path = 'path/to/test/image.jpg'
        self.model(img_path, result_path)
        self.assertTrue(os.path.exists(result_path))

    def test_process_directory(self):
        result_dir = 'test_results'
        img_dir = 'path/to/test/images'
        self.model(img_dir, result_dir)
        self.assertTrue(os.path.exists(result_dir))

    def test_invalid_image(self):
        invalid_img_path = 'path/to/invalid/image.jpg'
        result_path = 'test_invalid_result.png'
        with self.assertRaises(Exception):
            self.model(invalid_img_path, result_path)

if __name__ == '__main__':
    unittest.main()
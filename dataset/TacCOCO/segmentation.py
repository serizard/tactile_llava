import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from torch.multiprocessing import Process, Manager

def convert_mask_to_dict(mask_output):
    converted = {}
    for key, value in mask_output.items():
        if isinstance(value, np.ndarray):
            if key == 'segmentation':
                value = value.astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(value))
                rle['counts'] = rle['counts'].decode('utf-8')
                converted[key] = rle
            else:
                converted[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            converted[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            converted[key] = int(value)
        elif isinstance(value, list):
            # Handle lists that might contain NumPy types
            converted[key] = [
                float(x) if isinstance(x, (np.float32, np.float64)) else
                int(x) if isinstance(x, (np.int32, np.int64)) else x
                for x in value
            ]
        else:
            converted[key] = value
    return converted

# def convert_mask_to_dict(mask_output):
#     converted = {}
#     for key, value in mask_output.items():
#         if isinstance(value, np.ndarray):
#             if key == 'segmentation':
#                 value = value.astype(np.uint8)
#                 rle = mask_util.encode(np.asfortranarray(value))
#                 rle['counts'] = rle['counts'].decode('utf-8')
#                 converted[key] = rle
#             else:
#                 converted[key] = value.tolist()
#         else:
#             converted[key] = value
#     return converted

class MaskGenerator:
    def __init__(self, model_type, gpu_id, model_path):
        self.model_type = model_type
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        
        if model_type == 'sam2':
            self.model = SAM2AutomaticMaskGenerator.from_pretrained(model_path)
        elif model_type == 'yolov11':
            self.model = YOLO(model_path)
            self.model.to(f'cuda:{gpu_id}')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_masks(self, image):
        with torch.inference_mode():
            if self.model_type == 'sam2':
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.model.generate(image)
            else:
                results = self.model(image, verbose=False)[0]
                masks = []
                
                if results.masks is not None:
                    seg_masks = results.masks.data.cpu().numpy()
                    boxes = results.boxes.data.cpu().numpy()
                    
                    for i, (mask, box) in enumerate(zip(seg_masks, boxes)):
                        if mask.shape[:2] != image.shape[:2]:
                            mask = self.resize_mask(mask, image.shape[:2])
                        
                        mask_dict = {
                            'segmentation': mask.astype(bool),
                            'area': float(mask.sum()),
                            'bbox': box[:4].tolist(),  
                            'predicted_iou': float(box[4]),  
                            'point_coords': [(box[0] + box[2])/2, (box[1] + box[3])/2],
                            'stability_score': float(box[4]), 
                            'crop_box': box[:4].tolist(),
                            'category_id': int(box[5]) 
                        }
                        masks.append(mask_dict)
                
                return masks

    @staticmethod
    def resize_mask(mask, target_size):
        """Resize a mask to target size."""
        return np.array(Image.fromarray(mask).resize(
            (target_size[1], target_size[0]), 
            resample=Image.NEAREST
        ))

def process_image_batch(gpu_id, img_batch, coco_root, split, model_type, shared_dict, model_path):
    predictor = MaskGenerator(model_type, gpu_id, model_path)
    
    for img_info in tqdm(img_batch, desc=f'GPU {gpu_id} - {model_type}'):
        img_path = os.path.join(coco_root, split, img_info['file_name'])
        
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            masks = predictor.generate_masks(image)
            
            converted_masks = [convert_mask_to_dict(mask) for mask in masks]
            
            shared_dict[img_info['file_name']] = {
                'image_id': img_info['id'],
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'masks': converted_masks
            }
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            continue

def generate_masks_for_dataset(coco_root, gpu_ids, model_type, split='val2017', model_path=None):
    ann_file = os.path.join(coco_root, 'annotations', f'instances_{split}.json')
    coco = COCO(ann_file)
    
    all_imgs = coco.loadImgs(coco.getImgIds())
    
    n_gpus = len(gpu_ids)
    batch_size = len(all_imgs) // n_gpus
    img_batches = [all_imgs[i:i + batch_size] for i in range(0, len(all_imgs), batch_size)]
    
    if len(img_batches) > n_gpus:
        img_batches[n_gpus-1].extend(img_batches[n_gpus])
        img_batches = img_batches[:n_gpus]
    
    manager = Manager()
    shared_dict = manager.dict()
    processes = []
    
    for gpu_id, img_batch in zip(gpu_ids, img_batches):
        p = Process(target=process_image_batch, 
                   args=(gpu_id, img_batch, coco_root, split, model_type, shared_dict, model_path))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    return dict(shared_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True,
                        help='GPU IDs to use for distributed processing')
    parser.add_argument('--coco_root', type=str, default='/root/datasets/coco2017',
                        help='Path to COCO dataset root directory')
    parser.add_argument('--model_type', type=str, choices=['sam2', 'yolov11', 'both'],
                        default='both', help='Model to use for mask generation')
    parser.add_argument('--sam2_model', type=str, default='facebook/sam2-hiera-large',
                        help='Hugging Face model name for SAM2')
    parser.add_argument('--yolo_model', type=str, default='/root/workspace/ckpts/yolo11l-seg.pt',
                        help='Path to YOLO segmentation model')
    args = parser.parse_args()

    if args.model_type in ['sam2', 'both']:
        print("Generating masks using SAM2...")
        results_train_sam2 = generate_masks_for_dataset(args.coco_root, args.gpu_ids, 'sam2', 'train2017', args.sam2_model)
        results_val_sam2 = generate_masks_for_dataset(args.coco_root, args.gpu_ids, 'sam2', 'val2017', args.sam2_model)
        
        results_sam2 = {**results_train_sam2, **results_val_sam2}
        with open(os.path.dirname(__file__) + '/sam2_masks_coco.json', 'w', encoding='utf-8') as f:
            json.dump(results_sam2, f)
        print("SAM2 results saved to sam2_masks_coco.json")

    if args.model_type in ['yolov11', 'both']:
        print("Generating masks using YOLOv11...")
        results_train_yolo = generate_masks_for_dataset(args.coco_root, args.gpu_ids, 'yolov11', 'train2017', args.yolo_model)
        results_val_yolo = generate_masks_for_dataset(args.coco_root, args.gpu_ids, 'yolov11', 'val2017', args.yolo_model)
        
        results_yolo = {**results_train_yolo, **results_val_yolo}
        results_yolo = results_val_yolo
        with open(os.path.dirname(__file__) + '/yolov11_masks_coco.json', 'w', encoding='utf-8') as f:
            json.dump(results_yolo, f)
        print("YOLOv11 results saved to yolov11_masks_coco.json")


if __name__ == "__main__":
    main()
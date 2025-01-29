import os
import json
import uuid
import random
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import requests
from PIL import Image, ImageDraw
from dataclasses import dataclass
from dotenv import load_dotenv
from glob import glob
import argparse
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from tqdm import tqdm

import sys
sys.path.append('/root/workspace/tactile_llava')
from dataset.visual_prompt_generator import image_blending
from dataset.utils import mask_to_polygons

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class ConversationTurn:
    from_role: str
    value: str

@dataclass
class DatasetEntry:
    id: str
    image: str
    bboxes: List[List[float]]
    segmentations: List[List]
    answer: str

class TactileDatasetGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.system_message = open(CURRENT_DIR + "/system_message.txt", "r").read()
        self.prompt = open(CURRENT_DIR + "/prompt.txt", "r").read()

    def create_image_url(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    

    def query_gpt4v(self, original_image: Image.Image, marked_image: Image.Image) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.create_image_url(original_image)
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.create_image_url(marked_image)
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }

        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


    def generate_entry(self, image_path: str, bbox: List, segment: List) -> Optional[DatasetEntry]:
        # try:
        image = Image.open(image_path)
        bbox = [float(x) for x in bbox]
        segment = mask_util.decode(segment)
        marked_image = image_blending(image, shape = 'mask', bbox_coord = bbox, segmentation = segment, rgb_value = (255, 0, 255))
        response = ""
        for _ in range(args.max_retries):
            if len(response.split(',')) == 5:
                break
            response = self.query_gpt4v(image, marked_image)
        encoded_segment = mask_to_polygons(segment)

        return DatasetEntry(
            id=str(uuid.uuid4()),
            image=image_path,
            bboxes=[bbox],
            segmentations=[[encoded_segment]],
            answer=response
        )
        # except Exception as e:
        #     print(f"Error occurred while processing {image_path}: {e}")
        #     return None

    def generate_dataset(self, image_paths: List[str], bbox_data: List[str], segmentation_data: List[str], output_path: str) -> None:
        dataset = []
        
        with tqdm(total=len(image_paths)) as pbar:
            for i, image_path in enumerate(image_paths):
                entry = self.generate_entry(image_path, bbox_data[i], segmentation_data[i])
                if entry:
                    dataset.append(entry.__dict__)
                pbar.update(1)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)



def sample_segmentation_data(n_samples, data, threshold=None):
    areas = np.array([x['area'] for x in data])
    if threshold is not None:
        mask = areas > threshold
        data_array = areas[mask]
    else:
        data_array = areas
    mean = np.mean(data_array)
    std = np.std(data_array)
    probabilities = np.exp(-0.5 * ((data_array - mean) / std) ** 2)
    probabilities = probabilities / np.sum(probabilities) 
    
    selected_indices = np.random.choice(
        len(data_array), 
        size=n_samples, 
        replace=False,
        p=probabilities
    )

    return selected_indices


def check_split(data_path, file_name):
    train_path = f"{data_path}/train2017/{file_name}"
    if os.path.exists(train_path):
        return train_path
    else:
        return train_path.replace('train2017', 'val2017')

def main(args):
    generator = TactileDatasetGenerator(API_KEY)

    with open(args.segmentation_file, 'r') as f:
        segmentation_data = json.load(f)

    # sampling the segmentation data
    seg_only = []
    annotated_segs = {}
    idx = 0
    for v in segmentation_data.values():
        seg_only += [mask for mask in v['masks']]
        for mask in v['masks']:
            annotated_segs.update({idx: {'segmentation': mask['segmentation'],
                                         'fn': check_split(args.data_path, v['file_name']),
                                         'bbox': mask['bbox']}})
            idx += 1
    sampled_indices = sample_segmentation_data(args.n_samples, seg_only, args.size_threshold)
    sampled_segs = [annotated_segs[i] for i in sampled_indices]
    del seg_only, annotated_segs

    image_paths = [x['fn'] for x in sampled_segs]
    bbox_data = [x['bbox'] for x in sampled_segs]
    segmentation_data = [x['segmentation'] for x in sampled_segs]

    generator.generate_dataset(image_paths, bbox_data, segmentation_data, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_file", type=str, default="/root/workspace/tactile_llava/dataset/TacCOCO/yolov11_masks_coco.json")
    parser.add_argument("--data_path", type=str, default="/root/datasets/coco2017")
    parser.add_argument("--size_threshold", type=float, default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="/root/workspace/tactile_llava/dataset/gpt_instruction/vip-llava-tactile-task.json")
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()
    
    main(args)
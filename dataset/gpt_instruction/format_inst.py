import json
from PIL import Image
from tqdm import tqdm
import os

import sys
sys.path.append("/root/workspace/tactile_llava")
from dataset.visual_prompt_organizer import vip_processor

if __name__ == "__main__":
    # Load the data
    with open('/root/workspace/tactile_llava/dataset/gpt_instruction/vip-llava-tactile-task.json', 'r') as f:
        data = json.load(f)

    formatted_data = []
    for item in tqdm(data):
        if len(item['answer'].split(',')) != 5:
            continue

        image = Image.open(item['image']).convert('RGB')
        segmentation = item['segmentations']
        bbox = item['bboxes']

        # Create source dictionary with required format
        source = {
            'id': 'TacCOCO-task',
            'image': item['image'],
            'bboxes': bbox,
            'segmentations': segmentation,
            'answer': item['answer'],
        }
        data_args = {'alpha': 255}
        try:
            image, conv = vip_processor(source, image, 336, data_args)
        except Exception as e:
            print(segmentation)
            continue
        
        save_path = os.path.join(sys.path[-1], 'dataset/TacCOCO/images',os.path.basename(item['image']))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

        if 'mask' in conv[0]['value']:
            formatted_data.append({
                'id': item['id'],
                'image': save_path,
                'segmentations': segmentation,
                'conversation': conv
            })
        else:
            formatted_data.append({
                'id': item['id'],
                'image': item['image'],
                'bboxes': bbox,
                'conversation': conv
            })
            
       
    with open('/root/workspace/tactile_llava/dataset/gpt_instruction/vip-llava-tactile-task-formatted.json', 'w') as f:
        json.dump(formatted_data, f, indent=4)
        
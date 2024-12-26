import json
import asyncio
import aiohttp
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from pycocotools.coco import COCO
from tqdm import tqdm

@dataclass
class Config:
    api_key: str = "Your API key here"
    base_path: Path = Path('/root/workspace/tactile_llava/dataset')
    coco_path: Path = Path('/root/datasets/coco2017/annotations')
    batch_size: int = 10
    max_retries: int = 3
    
class COCODataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.system_prompts = self._load_system_prompts()
        self.fewshot_examples = self._load_fewshot_examples()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }
        
    def _load_system_prompts(self) -> str:
        with open(self.config.base_path / 'prompts/system.txt', 'r') as f:
            return f.read()
            
    def _load_fewshot_examples(self) -> List[Dict[str, str]]:
        examples = []
        for i in range(2):
            sample = {}
            for tp in ['caps', 'conv']:
                with open(self.config.base_path / f'prompts/{i}_{tp}.txt', 'r') as f:
                    sample[tp] = f.read()
            examples.append({
                'context': sample['caps'],
                'responses': sample['conv']
            })
        return examples
        
    def _prepare_messages(self, query_context: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompts}]
        
        for sample in self.fewshot_examples:
            messages.extend([
                {'role': 'user', 'content': sample['context']},
                {'role': 'assistant', 'content': sample['responses']}
            ])
        messages.append({'role': 'user', 'content': query_context})
        return messages

    async def _process_single_image(
        self, 
        session: aiohttp.ClientSession,
        image_id: int,
        captions_coco: COCO,
        bbox_coco: COCO
    ) -> Dict[str, Any]:
        try:
            ann_ids = captions_coco.getAnnIds(imgIds=image_id)
            annotations = captions_coco.loadAnns(ann_ids)
            captions = " ".join(ann['caption'] for ann in annotations)
            
            bbox_ann_ids = bbox_coco.getAnnIds(imgIds=image_id)
            bbox_annotations = bbox_coco.loadAnns(bbox_ann_ids)
            
            bbox_details = [
                f"{bbox_coco.loadCats(ann['category_id'])[0]['name']}({idx}): {ann['bbox']}"
                for idx, ann in enumerate(bbox_annotations)
            ]
            bbox_query = ", ".join(bbox_details)
            
            query_context = f"{captions}\n\n{bbox_query}"
            
            payload = {
                "model": "gpt-4o",
                "messages": self._prepare_messages(query_context),
                "max_tokens": 2000
            }
            
            for retry in range(self.config.max_retries):
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=self.headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            return {
                                'context': query_context,
                                'response': json.loads(response_data['choices'][0]['message']['content'])
                            }
                        await asyncio.sleep(1 * (retry + 1)) 
                except Exception as e:
                    if retry == self.config.max_retries - 1:
                        print(f"Failed to process image {image_id} after {self.config.max_retries} retries: {str(e)}")
                        return None
                    await asyncio.sleep(1 * (retry + 1))
                    
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            return None

    async def process_images(self, start_idx: int = 0, num_images: int = 50) -> Dict[int, Dict]:
        """Process multiple images concurrently"""
        captions_train = COCO(self.config.coco_path / 'captions_train2017.json')
        bbox_train = COCO(self.config.coco_path / 'instances_train2017.json')
        
        image_ids = captions_train.getImgIds()[start_idx:start_idx + num_images]
        
        async with aiohttp.ClientSession() as session:
            results = {}
            with tqdm(total=len(image_ids)) as pbar:
                for i in range(0, len(image_ids), self.config.batch_size):
                    batch_ids = image_ids[i:i + self.config.batch_size]
                    tasks = [
                        self._process_single_image(session, img_id, captions_train, bbox_train)
                        for img_id in batch_ids
                    ]
                    
                    batch_results = await tqdm_asyncio.gather(*tasks)
                    
                    results.update({
                        img_id: result
                        for img_id, result in zip(batch_ids, batch_results)
                        if result is not None
                    })
                    
                    await asyncio.sleep(0.5)
                    pbar.update(len(batch_ids))
                
        return results

async def main():
    config = Config()
    processor = COCODataProcessor(config)
    
    results = await processor.process_images()
    
    output_path = config.base_path / 'instructions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    asyncio.run(main())
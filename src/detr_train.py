
import sys
sys.stdout = open('/home/jupyter-st126222/Project/rdd2022/logs/detr_full_train.log', 'w', buffering=1)
sys.stderr = sys.stdout

from pathlib import Path
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import torch
import torch.nn.functional as F
import time
from torch.optim import AdamW

BASE_DIR = Path('/home/jupyter-st126222/Project/rdd2022')
FILTERED_DIR = BASE_DIR / 'data' / 'RDD_FILTERED'
COCO_DIR = BASE_DIR / 'data' / 'RDD_COCO'
SAVED_DIR = BASE_DIR / 'saved'
NUM_CLASSES = 4
NUM_EPOCHS = 20
BATCH_SIZE = 4
MAX_GRAD_NORM = 0.1
device = torch.device('cuda:1')

processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

class RDDCocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, processor):
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_file))
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.processor = processor
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        target = {'image_id': img_id, 'annotations': anns}
        encoding = self.processor(images=image, annotations=target, return_tensors='pt')
        return encoding['pixel_values'].squeeze(), encoding['labels'][0]

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_h = max(img.shape[1] for img in pixel_values)
    max_w = max(img.shape[2] for img in pixel_values)
    padded_imgs, pixel_masks = [], []
    for img in pixel_values:
        c, h, w = img.shape
        padded = F.pad(img, (0, max_w-w, 0, max_h-h), value=0)
        padded_imgs.append(padded)
        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        pixel_masks.append(mask)
    return {'pixel_values': torch.stack(padded_imgs), 'pixel_mask': torch.stack(pixel_masks), 'labels': labels}

train_dataset = RDDCocoDataset(FILTERED_DIR/'train'/'images', COCO_DIR/'train_annotations.json', processor)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

checkpoint_path = SAVED_DIR / 'detr_rdd2022_full'
checkpoint_path.mkdir(parents=True, exist_ok=True)

print(f'Starting training: {len(train_dataset)} images, {NUM_EPOCHS} epochs', flush=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    for i, batch in enumerate(train_dataloader):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 1000 == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Step {i}/{len(train_dataloader)} | Loss: {loss.item():.4f}', flush=True)
    avg_loss = epoch_loss / len(train_dataloader)
    print(f'Epoch {epoch+1} done | Avg loss: {avg_loss:.4f} | Time: {(time.time()-epoch_start)/60:.1f} min', flush=True)
    model.save_pretrained(checkpoint_path / f'epoch_{epoch+1}')
    print(f'Checkpoint saved: epoch_{epoch+1}', flush=True)

print('Training complete!', flush=True)

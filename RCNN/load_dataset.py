import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as transforms

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        
        # Lấy tất cả image IDs
        self.image_ids = list(self.coco.imgs.keys())
        
        # Lấy category IDs và tạo mapping
        self.category_ids = sorted(self.coco.getCatIds())
        self.category_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}
        self.idx_to_category = {idx: cat_id for cat_id, idx in self.category_to_idx.items()}
        
        # Số lượng classes (+ 1 cho background)
        self.num_classes = len(self.category_ids) + 1
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Lấy annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Tạo targets
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:  # width > 0 and height > 0
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.category_to_idx[ann['category_id']] + 1)  # +1 vì 0 là background
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            # Trường hợp không có annotation
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def get_class_names(self):
        """Trả về tên các classes"""
        class_names = ['__background__']
        for cat_id in self.category_ids:
            cat_info = self.coco.cats[cat_id]
            class_names.append(cat_info['name'])
        return class_names

def get_transform(train=True):
    """Tạo transform cho dữ liệu"""
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))

def get_subset_indices(dataset, percent=0.1):
    from collections import defaultdict
    
    # Tạo dictionary để lưu indices theo từng category
    category_indices = defaultdict(list)
    
    # Duyệt qua tất cả images để group theo category
    for idx in range(len(dataset)):
        img_id = dataset.image_ids[idx]
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        
        # Lấy tất cả category_ids có trong image này
        categories_in_image = set()
        for ann in anns:
            if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:  # valid bbox
                categories_in_image.add(ann['category_id'])
        
        # Thêm index này vào tất cả categories có trong image
        for cat_id in categories_in_image:
            category_indices[cat_id].append(idx)
    
    # Lấy mẫu từ mỗi category
    selected_indices = set()
    
    print(f"Sampling {percent*100}% from each category:")
    for cat_id, indices in category_indices.items():
        cat_info = dataset.coco.cats[cat_id]
        cat_name = cat_info['name']
        
        # Tính số lượng mẫu cần lấy cho category này
        total_samples = len(indices)
        subset_size = max(1, int(total_samples * percent))  # Ít nhất 1 mẫu
        
        # Random sampling trong category này
        if subset_size >= total_samples:
            sampled_indices = indices
        else:
            sampled_indices = np.random.choice(indices, size=subset_size, replace=False)
        
        # Thêm vào tập kết quả
        selected_indices.update(sampled_indices)
        
        print(f"  {cat_name}: {subset_size}/{total_samples} samples")
    
    # Convert về list và sort
    final_indices = sorted(list(selected_indices))
    
    return final_indices

def load_coco_dataset(data_dir="coco_data", train_set_percent=0.1, batch_size=4, num_workers=4):
    """
    Load COCO dataset và chia thành train/test với tỉ lệ 70:30
    """
    # Đường dẫn đến annotations
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    
    # Đường dẫn đến images
    train_img_dir = os.path.join(data_dir, 'train2017')
    val_img_dir = os.path.join(data_dir, 'val2017')
    
    # Tạo dataset cho train và validation
    train_dataset = COCODataset(train_img_dir, train_ann_file, transform=get_transform(train=True))
    val_dataset = COCODataset(val_img_dir, val_ann_file, transform=get_transform(train=False))

    train_indices = get_subset_indices(train_dataset, train_set_percent)
    val_indices = get_subset_indices(val_dataset, train_set_percent)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Lấy thông tin về classes
    class_names = train_dataset.get_class_names()
    num_classes = train_dataset.num_classes
    
    print(f"Dataset loaded successfully!")
    print(f"Train images: {len(train_indices)}")
    print(f"Test images: {len(val_indices)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names[:10]}...")  # Hiển thị 10 classes đầu tiên
    
    return train_loader, test_loader, num_classes, class_names

if __name__ == "__main__":
    # Test loading dataset
    try:
        train_loader, test_loader, num_classes, class_names = load_coco_dataset(
            data_dir="coco_data",
            train_set_percent=0.1,
            batch_size=2,
            num_workers=0
        )
        
        print("\nTesting data loading...")
        
        # Test train loader
        for i, (images, targets) in enumerate(train_loader):
            print(f"Train batch {i+1}: {len(images)} images")
            for j, target in enumerate(targets):
                print(f"  Image {j+1}: {len(target['boxes'])} objects")
            if i >= 2:  # Chỉ test 3 batch đầu
                break
        
        # Test test loader
        for i, (images, targets) in enumerate(test_loader):
            print(f"Test batch {i+1}: {len(images)} images")
            if i >= 2:  # Chỉ test 3 batch đầu
                break
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have run download_dataset.sh first!")
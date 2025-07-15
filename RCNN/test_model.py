import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import json
from collections import defaultdict
import time

from model import create_model
from load_dataset import load_coco_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class FasterRCNNEvaluator:
    """
    Improved Evaluator cho Faster R-CNN với COCO metrics
    """
    def __init__(self, model, test_loader, device, class_names, coco_gt=None, confidence_threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.num_classes = len(class_names)
        self.coco_gt = coco_gt  # COCO ground truth API
        
        # Results storage
        self.coco_results = []
        self.evaluation_results = {}
        
    def evaluate_model(self):
        """
        Evaluate model với COCO metrics
        """
        print("Evaluating model with COCO metrics...")
        
        self.model.eval()
        total_images = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.test_loader),
                total=len(self.test_loader),
                desc='Evaluating'
            )
            
            for batch_idx, (images, targets) in progress_bar:
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                predictions = self.model(images)
                
                # Process each image in batch
                for img_idx, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
                    total_images += 1
                    
                    # Get image ID từ target
                    image_id = target.get('image_id', batch_idx * len(images) + img_idx)
                    if torch.is_tensor(image_id):
                        image_id = image_id.item()
                    
                    # Filter predictions by confidence threshold
                    keep_indices = prediction['scores'] > self.confidence_threshold
                    pred_boxes = prediction['boxes'][keep_indices]
                    pred_labels = prediction['labels'][keep_indices]
                    pred_scores = prediction['scores'][keep_indices]
                    
                    # Convert to COCO format
                    self.add_coco_predictions(image_id, pred_boxes, pred_labels, pred_scores)
                
                progress_bar.set_postfix({'Images': total_images})
        
        # Evaluate using COCO API
        if self.coco_gt is not None:
            self.evaluation_results = self.evaluate_with_coco_api()
        else:
            print("Warning: No COCO ground truth provided. Using basic evaluation.")
            self.evaluation_results = self.basic_evaluation()
        
        print(f"\nEvaluation completed on {total_images} images!")
        return self.evaluation_results
    
    def add_coco_predictions(self, image_id, pred_boxes, pred_labels, pred_scores):
        """
        Add predictions in COCO format
        """
        if len(pred_boxes) == 0:
            return
        
        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            # Convert từ (x1, y1, x2, y2) sang (x, y, width, height)
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            self.coco_results.append({
                'image_id': int(image_id),
                'category_id': int(label),
                'bbox': bbox,
                'score': float(score)
            })
    
    def evaluate_with_coco_api(self):
        """
        Evaluate using official COCO API
        """
        if len(self.coco_results) == 0:
            print("No predictions to evaluate!")
            return {}
        
        # Create COCO results object
        coco_dt = self.coco_gt.loadRes(self.coco_results)
        
        # Initialize COCO evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract results
        results = {
            'mAP': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # AP @ IoU=0.50
            'mAP_75': coco_eval.stats[2],  # AP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # AP for small objects
            'mAP_medium': coco_eval.stats[4],  # AP for medium objects
            'mAP_large': coco_eval.stats[5],  # AP for large objects
            'mAR_1': coco_eval.stats[6],  # AR given 1 detection per image
            'mAR_10': coco_eval.stats[7],  # AR given 10 detections per image
            'mAR_100': coco_eval.stats[8],  # AR given 100 detections per image
            'mAR_small': coco_eval.stats[9],  # AR for small objects
            'mAR_medium': coco_eval.stats[10],  # AR for medium objects
            'mAR_large': coco_eval.stats[11],  # AR for large objects
        }
        
        return results
    
    def basic_evaluation(self):
        """
        Basic evaluation without COCO API (fallback)
        """
        print("Performing basic evaluation...")
        
        # Implement basic precision/recall như code cũ
        # (Giữ nguyên logic từ code gốc của bạn)
        
        return {
            'note': 'Basic evaluation - for full COCO metrics, provide COCO ground truth API'
        }
    
    def print_evaluation_results(self):
        """
        Print detailed evaluation results
        """
        if not self.evaluation_results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("COCO EVALUATION RESULTS")
        print("="*60)
        
        if 'mAP' in self.evaluation_results:
            results = self.evaluation_results
            print(f"\nObject Detection Metrics:")
            print(f"  mAP @ IoU=0.50:0.95: {results['mAP']:.4f}")
            print(f"  mAP @ IoU=0.50:     {results['mAP_50']:.4f}")
            print(f"  mAP @ IoU=0.75:     {results['mAP_75']:.4f}")
            
            print(f"\nObject Size Breakdown:")
            print(f"  mAP (small):        {results['mAP_small']:.4f}")
            print(f"  mAP (medium):       {results['mAP_medium']:.4f}")
            print(f"  mAP (large):        {results['mAP_large']:.4f}")
            
            print(f"\nRecall Metrics:")
            print(f"  mAR @ 1 det/img:    {results['mAR_1']:.4f}")
            print(f"  mAR @ 10 det/img:   {results['mAR_10']:.4f}")
            print(f"  mAR @ 100 det/img:  {results['mAR_100']:.4f}")
            
            print(f"\nRecall by Object Size:")
            print(f"  mAR (small):        {results['mAR_small']:.4f}")
            print(f"  mAR (medium):       {results['mAR_medium']:.4f}")
            print(f"  mAR (large):        {results['mAR_large']:.4f}")
        else:
            print("Basic evaluation results:")
            for key, value in self.evaluation_results.items():
                print(f"  {key}: {value}")
    
    def calculate_per_class_ap(self):
        """
        Calculate Average Precision per class
        """
        if self.coco_gt is None:
            print("COCO ground truth not available for per-class AP calculation.")
            return {}
        
        # Create COCO results object
        coco_dt = self.coco_gt.loadRes(self.coco_results)
        
        # Initialize COCO evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Extract per-class AP
        per_class_ap = {}
        
        # Get category info
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        
        for cat in cats:
            cat_id = cat['id']
            cat_name = cat['name']
            
            # Get AP for this category
            # coco_eval.eval['precision'] shape: [T, R, K, A, M]
            # T: IoU thresholds, R: recall thresholds, K: categories, A: areas, M: max dets
            precision = coco_eval.eval['precision']
            
            if precision.size > 0:
                # AP @ IoU=0.50:0.95, all areas, max_dets=100
                ap = np.mean(precision[:, :, cat_id-1, 0, 2])  # -1 because cat_id starts from 1
                per_class_ap[cat_name] = ap
        
        return per_class_ap
    
    def save_results(self, save_path):
        """
        Save evaluation results
        """
        results_to_save = {
            'evaluation_results': self.evaluation_results,
            'per_class_ap': self.calculate_per_class_ap() if self.coco_gt else {},
            'total_predictions': len(self.coco_results),
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Results saved to {save_path}")
    
    def visualize_predictions(self, num_images=5, save_dir='visualizations'):
        """
        Visualize predictions on sample images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get sample images from test loader
        sample_images = []
        sample_targets = []
        
        for i, (images, targets) in enumerate(self.test_loader):
            sample_images.extend(images[:num_images])
            sample_targets.extend(targets[:num_images])
            if len(sample_images) >= num_images:
                break
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            device_images = [img.to(self.device) for img in sample_images[:num_images]]
            predictions = self.model(device_images)
        
        # Visualize each image
        for i in range(min(num_images, len(sample_images))):
            self.visualize_single_image(
                sample_images[i], 
                sample_targets[i], 
                predictions[i],
                save_path=os.path.join(save_dir, f'prediction_{i+1}.png')
            )
    
    def visualize_single_image(self, image, target, prediction, save_path):
        """
        Visualize single image with ground truth and predictions
        """
        # Convert tensor to numpy
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Convert to uint8
        img_np = (img_np * 255).astype(np.uint8)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot ground truth
        ax1.imshow(img_np)
        ax1.set_title('Ground Truth')
        ax1.axis('off')
        
        # Draw ground truth boxes
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            ax1.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='green', linewidth=2))
            class_name = self.class_names[label] if label < len(self.class_names) else f'class_{label}'
            ax1.text(x1, y1-5, class_name, color='green', fontsize=8)
        
        # Plot predictions
        ax2.imshow(img_np)
        ax2.set_title('Predictions')
        ax2.axis('off')
        
        # Draw prediction boxes
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        keep_indices = pred_scores > self.confidence_threshold
        pred_boxes = pred_boxes[keep_indices]
        pred_labels = pred_labels[keep_indices]
        pred_scores = pred_scores[keep_indices]
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box
            ax2.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='red', linewidth=2))
            class_name = self.class_names[label] if label < len(self.class_names) else f'class_{label}'
            ax2.text(x1, y1-5, f'{class_name}: {score:.2f}', color='red', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def load_trained_model(model_path, num_classes, device):
    """
    Load trained model from checkpoint
    """
    model = create_model(num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def test_model(dataset_path, annotations_path=None):
    """
    Main evaluation function with COCO API support
    """
    print("=== Faster R-CNN COCO Evaluation ===")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading COCO dataset...")
    try:
        train_loader, test_loader, num_classes, class_names = load_coco_dataset(
            data_dir=dataset_path,
            train_set_percent=1,
            batch_size=4,
            num_workers=2
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load COCO ground truth API
    coco_gt = None
    if annotations_path and os.path.exists(annotations_path):
        print("Loading COCO ground truth API...")
        coco_gt = COCO(annotations_path)
        print(f"Loaded {len(coco_gt.getImgIds())} images from COCO annotations")
    else:
        print("Warning: COCO annotations not found. Using basic evaluation.")
    
    # Load trained model
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first.")
        return
    
    print("Loading trained model...")
    model = load_trained_model(model_path, num_classes, device)
    
    # Create evaluator
    evaluator = FasterRCNNEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        coco_gt=coco_gt,
        confidence_threshold=0.5
    )
    
    # Evaluate model
    start_time = time.time()
    results = evaluator.evaluate_model()
    eval_time = time.time() - start_time
    
    # Print results
    evaluator.print_evaluation_results()
    
    print(f"\nEvaluation time: {eval_time:.2f} seconds")
    print(f"Average time per image: {eval_time/len(test_loader.dataset):.4f} seconds")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    evaluator.save_results('results/coco_evaluation_results.json')
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    evaluator.visualize_predictions(num_images=5, save_dir='results/visualizations')
    
    print("\nEvaluation completed!")
    print(f"Results saved in 'results/' directory")
    
    # Return main mAP metric
    return results.get('mAP', 0)
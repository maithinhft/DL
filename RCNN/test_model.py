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
    Evaluator class cho Faster R-CNN
    """
    def __init__(self, model, test_loader, device, class_names, confidence_threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.num_classes = len(class_names)
        
        # Statistics
        self.predictions = []
        self.ground_truths = []
        self.evaluation_results = {}
        
    def evaluate_model(self):
        """
        Evaluate model trên test dataset
        """
        print("Evaluating model...")
        
        self.model.eval()
        total_images = 0
        total_predictions = 0
        total_ground_truths = 0
        
        # Statistics per class
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})
        
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
                    
                    # Filter predictions by confidence threshold
                    keep_indices = prediction['scores'] > self.confidence_threshold
                    pred_boxes = prediction['boxes'][keep_indices]
                    pred_labels = prediction['labels'][keep_indices]
                    pred_scores = prediction['scores'][keep_indices]
                    
                    # Ground truth
                    gt_boxes = target['boxes']
                    gt_labels = target['labels']
                    
                    total_predictions += len(pred_boxes)
                    total_ground_truths += len(gt_boxes)
                    
                    # Calculate IoU and match predictions to ground truths
                    matches = self.match_predictions_to_ground_truth(
                        pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels
                    )
                    
                    # Update class statistics
                    self.update_class_statistics(matches, class_stats)
                    
                    # Store for detailed analysis
                    self.predictions.append({
                        'image_id': batch_idx * len(images) + img_idx,
                        'boxes': pred_boxes.cpu().numpy(),
                        'labels': pred_labels.cpu().numpy(),
                        'scores': pred_scores.cpu().numpy()
                    })
                    
                    self.ground_truths.append({
                        'image_id': batch_idx * len(images) + img_idx,
                        'boxes': gt_boxes.cpu().numpy(),
                        'labels': gt_labels.cpu().numpy()
                    })
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Images': total_images,
                    'Predictions': total_predictions,
                    'Ground Truths': total_ground_truths
                })
        
        # Calculate metrics
        self.evaluation_results = self.calculate_metrics(class_stats)
        
        print(f"\nEvaluation completed!")
        print(f"Total images: {total_images}")
        print(f"Total predictions: {total_predictions}")
        print(f"Total ground truths: {total_ground_truths}")
        
        return self.evaluation_results
    
    def match_predictions_to_ground_truth(self, pred_boxes, pred_labels, pred_scores, 
                                        gt_boxes, gt_labels, iou_threshold=0.5):
        """
        Match predictions to ground truth boxes based on IoU
        """
        matches = []
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return matches
        
        # Calculate IoU matrix
        iou_matrix = self.calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # Find best matches
        used_gt_indices = set()
        
        for pred_idx in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in used_gt_indices:
                    continue
                
                if (iou_matrix[pred_idx, gt_idx] > best_iou and 
                    pred_labels[pred_idx] == gt_labels[gt_idx]):
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'label': pred_labels[pred_idx].item(),
                    'score': pred_scores[pred_idx].item(),
                    'type': 'tp'  # True Positive
                })
                used_gt_indices.add(best_gt_idx)
            else:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': -1,
                    'iou': 0,
                    'label': pred_labels[pred_idx].item(),
                    'score': pred_scores[pred_idx].item(),
                    'type': 'fp'  # False Positive
                })
        
        # Add false negatives (unmatched ground truths)
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in used_gt_indices:
                matches.append({
                    'pred_idx': -1,
                    'gt_idx': gt_idx,
                    'iou': 0,
                    'label': gt_labels[gt_idx].item(),
                    'score': 0,
                    'type': 'fn'  # False Negative
                })
        
        return matches
    
    def calculate_iou_matrix(self, pred_boxes, gt_boxes):
        """
        Calculate IoU matrix between predicted and ground truth boxes
        """
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou_matrix[pred_idx, gt_idx] = self.calculate_iou(pred_box, gt_box)
        
        return iou_matrix
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        """
        # Convert to numpy if tensor
        if torch.is_tensor(box1):
            box1 = box1.cpu().numpy()
        if torch.is_tensor(box2):
            box2 = box2.cpu().numpy()
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_class_statistics(self, matches, class_stats):
        """
        Update per-class statistics
        """
        for match in matches:
            label = match['label']
            match_type = match['type']
            
            if match_type == 'tp':
                class_stats[label]['tp'] += 1
            elif match_type == 'fp':
                class_stats[label]['fp'] += 1
            elif match_type == 'fn':
                class_stats[label]['fn'] += 1
                class_stats[label]['total_gt'] += 1
        
        # Count total ground truths
        for match in matches:
            if match['type'] in ['tp', 'fn']:
                class_stats[match['label']]['total_gt'] += 1
    
    def calculate_metrics(self, class_stats):
        """
        Calculate evaluation metrics
        """
        results = {
            'per_class': {},
            'overall': {}
        }
        
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        overall_gt = 0
        
        # Calculate per-class metrics
        for class_idx, stats in class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            total_gt = stats['total_gt']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{class_idx}'
            
            results['per_class'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'total_gt': total_gt
            }
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            overall_gt += total_gt
        
        # Calculate overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_accuracy = overall_tp / overall_gt if overall_gt > 0 else 0
        
        results['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'accuracy': overall_accuracy,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn,
            'total_gt': overall_gt
        }
        
        return results
    
    def print_evaluation_results(self):
        """
        Print detailed evaluation results
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_model() first.")
            return
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Overall metrics
        overall = self.evaluation_results['overall']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall: {overall['recall']:.4f}")
        print(f"  F1-Score: {overall['f1_score']:.4f}")
        print(f"  True Positives: {overall['tp']}")
        print(f"  False Positives: {overall['fp']}")
        print(f"  False Negatives: {overall['fn']}")
        print(f"  Total Ground Truths: {overall['total_gt']}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        print("-" * 80)
        
        for class_name, metrics in self.evaluation_results['per_class'].items():
            print(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} {metrics['tp']:<5} {metrics['fp']:<5} {metrics['fn']:<5}")
    
    def save_results(self, save_path):
        """
        Save evaluation results to JSON file
        """
        with open(save_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
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
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def test_model():
    """
    Main evaluation function
    """
    print("=== Faster R-CNN Evaluation ===")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading COCO dataset...")
    try:
        train_loader, test_loader, num_classes, class_names = load_coco_dataset(
            data_dir="coco_data",
            train_split=0.7,
            batch_size=4,
            num_workers=2
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load trained model
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using training_model.py")
        return
    
    print("Loading trained model...")
    model = load_trained_model(model_path, num_classes, device)
    
    # Create evaluator
    evaluator = FasterRCNNEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
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
    evaluator.save_results('results/evaluation_results.json')
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    evaluator.visualize_predictions(num_images=5, save_dir='results/visualizations')
    
    print("\nEvaluation completed!")
    print(f"Results saved in 'results/' directory")
    
    # Return main accuracy metric
    return results['overall']['accuracy']
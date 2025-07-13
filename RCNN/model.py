import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import warnings
warnings.filterwarnings("ignore")

class FasterRCNN(nn.Module):
    """
    Faster R-CNN model cho object detection
    """
    def __init__(self, num_classes, pretrained=True):
        super(FasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            progress=True,
            num_classes=91,  # COCO has 91 classes (including background)
            pretrained_backbone=True
        )
        
        # Thay đổi classifier head để phù hợp với số classes của chúng ta
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Có thể customize RPN nếu cần
        # self.customize_rpn()
        
    def customize_rpn(self):
        """
        Customize Region Proposal Network nếu cần
        """
        # Tạo anchor generator với các kích thước và tỷ lệ khác nhau
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        # Thay đổi anchor generator
        self.model.rpn.anchor_generator = anchor_generator
        
        # Cập nhật số lượng anchors cho RPN head
        self.model.rpn.head.conv = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1
        )
        
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        self.model.rpn.head.cls_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
        self.model.rpn.head.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of images (tensors)
            targets: List of targets (dicts) - chỉ dùng khi training
        
        Returns:
            losses (dict) khi training
            predictions (list) khi inference
        """
        return self.model(images, targets)
    
    def predict(self, images, confidence_threshold=0.5):
        """
        Predict objects in images
        
        Args:
            images: List of images hoặc single image tensor
            confidence_threshold: Ngưỡng confidence để lọc predictions
            
        Returns:
            predictions: List of predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                images = [images]
            
            predictions = self.model(images)
            
            # Lọc predictions dựa trên confidence threshold
            filtered_predictions = []
            for pred in predictions:
                # Lọc boxes có confidence > threshold
                keep_indices = pred['scores'] > confidence_threshold
                
                filtered_pred = {
                    'boxes': pred['boxes'][keep_indices],
                    'labels': pred['labels'][keep_indices],
                    'scores': pred['scores'][keep_indices]
                }
                filtered_predictions.append(filtered_pred)
            
            return filtered_predictions
    
    def get_model_summary(self):
        """
        In thông tin về model
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model Summary:")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        print(f"- Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"- Number of classes: {self.num_classes}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_classes': self.num_classes
        }
    
    def freeze_backbone(self):
        """
        Freeze backbone layers để fine-tuning
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print("Backbone layers frozen for fine-tuning")
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone layers
        """
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        print("Backbone layers unfrozen")
    
    def save_model(self, path):
        """
        Lưu model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

def create_model(num_classes, pretrained=True):
    """
    Factory function để tạo model
    
    Args:
        num_classes: Số lượng classes (bao gồm background)
        pretrained: Sử dụng pre-trained weights hay không
    
    Returns:
        model: Faster R-CNN model
    """
    model = FasterRCNN(num_classes=num_classes, pretrained=pretrained)
    return model

# Custom loss function nếu cần
class FasterRCNNLoss(nn.Module):
    """
    Custom loss function cho Faster R-CNN
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(FasterRCNNLoss, self).__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for regression loss
    
    def forward(self, loss_dict):
        """
        Tính tổng loss từ các component losses
        
        Args:
            loss_dict: Dictionary chứa các losses từ model
            
        Returns:
            total_loss: Tổng loss
        """
        # Faster R-CNN trả về các losses:
        # - loss_classifier: Classification loss
        # - loss_box_reg: Box regression loss
        # - loss_rpn_box_reg: RPN box regression loss
        # - loss_objectness: RPN objectness loss
        
        total_loss = (
            loss_dict['loss_classifier'] * self.alpha +
            loss_dict['loss_box_reg'] * self.beta +
            loss_dict['loss_rpn_box_reg'] +
            loss_dict['loss_objectness']
        )
        
        return total_loss

def demo_model():
    # Test model creation
    print("Testing Faster R-CNN model creation...")
    
    # Tạo model với 91 classes (COCO)
    model = create_model(num_classes=91, pretrained=True)
    
    # In thông tin model
    model_info = model.get_model_summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Tạo dummy input
    dummy_images = [torch.randn(3, 416, 416) for _ in range(2)]
    
    # Test inference mode
    model.model.eval()
    with torch.no_grad():
        predictions = model.predict(dummy_images, confidence_threshold=0.5)
        print(f"Predictions: {len(predictions)} images")
        for i, pred in enumerate(predictions):
            print(f"  Image {i+1}: {len(pred['boxes'])} objects detected")
    
    print("Model test completed successfully!")
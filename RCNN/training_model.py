import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from model import create_model, FasterRCNNLoss
from load_dataset import load_coco_dataset

class FasterRCNNTrainer:
    """
    Trainer class cho Faster R-CNN
    """
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Setup loss function
        self.criterion = FasterRCNNLoss(alpha=1.0, beta=1.0)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
    def train_one_epoch(self, epoch):
        """
        Train model for one epoch
        """
        self.model.train()
        
        total_loss = 0
        loss_components = defaultdict(float)
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            enumerate(self.train_loader), 
            total=num_batches,
            desc=f'Epoch {epoch+1}/{self.config["epochs"]}'
        )
        
        for batch_idx, (images, targets) in progress_bar:
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            total_batch_loss = self.criterion(loss_dict)
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            if self.config['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['clip_grad_norm']
                )
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += total_batch_loss.item()
            for key, value in loss_dict.items():
                loss_components[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config['log_interval'] == 0:
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', total_batch_loss.item(), step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', value.item(), step)
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_loss_components
    
    def validate(self, epoch):
        """
        Validate model
        """
        self.model.eval()
        
        total_loss = 0
        loss_components = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=num_batches,
                desc=f'Validation Epoch {epoch+1}'
            )
            
            for batch_idx, (images, targets) in progress_bar:
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                
                # Calculate total loss
                total_batch_loss = self.criterion(loss_dict)
                
                # Update statistics
                total_loss += total_batch_loss.item()
                for key, value in loss_dict.items():
                    loss_components[key] += value.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Val Loss': f'{total_batch_loss.item():.4f}',
                    'Avg Val Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_loss_components
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f'Best model saved with validation loss: {loss:.4f}')
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f'Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch}')
        
        return start_epoch
    
    def plot_training_history(self):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['checkpoint_dir'], 'training_history.png'))
        plt.show()
    
    def train(self, start_epoch=0):
        """
        Main training loop
        """
        print(f"Starting training from epoch {start_epoch+1}")
        print(f"Training on device: {self.device}")
        print(f"Number of training batches: {len(self.train_loader)}")
        print(f"Number of validation batches: {len(self.val_loader)}")
        
        for epoch in range(start_epoch, self.config['epochs']):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_loss_components = self.train_one_epoch(epoch)
            
            # Validation phase
            val_loss, val_loss_components = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Epoch_Loss', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for key, value in train_loss_components.items():
                self.writer.add_scalar(f'Train/Epoch_{key}', value, epoch)
            for key, value in val_loss_components.items():
                self.writer.add_scalar(f'Val/Epoch_{key}', value, epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]} Summary:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Learning Rate: {current_lr:.2e}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # Print detailed loss components
            print('  Train Loss Components:')
            for key, value in train_loss_components.items():
                print(f'    {key}: {value:.4f}')
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.config['early_stopping_patience'] > 0:
                if len(self.val_losses) >= self.config['early_stopping_patience']:
                    recent_losses = self.val_losses[-self.config['early_stopping_patience']:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        break
        
        print('Training completed!')
        print(f'Best validation loss: {self.best_val_loss:.4f}')
        
        # Close tensorboard writer
        self.writer.close()
        
        # Plot training history
        self.plot_training_history()
        
        return self.best_val_loss

def get_training_config():
    """
    Get default training configuration
    """
    config = {
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'epochs': 50,
        'batch_size': 4,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'clip_grad_norm': 0.5,
        'log_interval': 100,
        'save_interval': 5,
        'early_stopping_patience': 10,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'num_workers': 4
    }
    return config

def training_model():
    """
    Main training function
    """
    print("=== Faster R-CNN Training ===")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = get_training_config()
    
    # Load dataset
    print("Loading COCO dataset...")
    try:
        train_loader, val_loader, num_classes, class_names = load_coco_dataset(
            data_dir="coco_data",
            train_split=0.7,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have run download_dataset.sh first!")
        return
    
    # Create model
    print("Creating Faster R-CNN model...")
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Print model summary
    model_info = model.get_model_summary()
    
    # Optional: Freeze backbone for fine-tuning
    if config.get('freeze_backbone', False):
        model.freeze_backbone()
    
    # Create trainer
    trainer = FasterRCNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Start training
    start_epoch = 0
    
    # Resume from checkpoint if exists
    resume_checkpoint = config.get('resume_checkpoint', None)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch = trainer.load_checkpoint(resume_checkpoint)
    
    # Train the model
    best_val_loss = trainer.train(start_epoch=start_epoch)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {trainer.best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    model.save_model(final_model_path)
    
    print(f"Final model saved at: {final_model_path}")
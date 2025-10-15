"""
Model module for COMET-SEE project.

This module handles:
- Dataset creation
- Model architecture
- Training loops
- Evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import numpy as np
import cv2
from typing import List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DifferenceImageDataset(Dataset):
    """Dataset for difference images with binary classification."""
    
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        img_size: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            images: List of 2D numpy arrays (difference images)
            labels: List of binary labels (0=background, 1=comet)
            transform: Optional torchvision transforms
            img_size: Target image size for resizing
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single item from the dataset."""
        img = self.images[idx]
        label = self.labels[idx]
        
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert grayscale to RGB (3 channels)
        img_rgb = np.stack([img, img, img], axis=0)
        img_tensor = torch.FloatTensor(img_rgb)
        
        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label


class CometClassifier:
    """EfficientNet-based binary classifier for comet detection."""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 2,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of timm model to use
            num_classes: Number of output classes
            device: Device to use (None = auto-detect)
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load pretrained model
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized {model_name} on {self.device}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_transforms(self, augment: bool = False) -> transforms.Compose:
        """
        Get image transforms for preprocessing.
        
        Args:
            augment: Whether to include data augmentation
            
        Returns:
            Composed transforms
        """
        if augment:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, List[int], List[int]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy, predictions, labels)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        learning_rate: float = 0.0001,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save best model (optional)
            
        Returns:
            Dictionary with training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(
                val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        logger.info(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")
        return history
    
    def evaluate(
        self,
        val_loader: DataLoader,
        class_names: List[str] = ['Background', 'Comet']
    ) -> dict:
        """
        Evaluate model and return metrics.
        
        Args:
            val_loader: Validation data loader
            class_names: Names of classes for report
            
        Returns:
            Dictionary with evaluation metrics
        """
        criterion = nn.CrossEntropyLoss()
        _, accuracy, predictions, labels = self.validate(val_loader, criterion)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, 
            predictions,
            target_names=class_names,
            output_dict=True
        )
        
        logger.info("\nConfusion Matrix:")
        logger.info(cm)
        logger.info("\nClassification Report:")
        logger.info(classification_report(labels, predictions, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'report': report,
            'predictions': predictions,
            'labels': labels
        }
    
    def load(self, model_path: str) -> None:
        """Load model weights from file."""
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        logger.info(f"Loaded model from {model_path}")
    
    def save(self, model_path: str) -> None:
        """Save model weights to file."""
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
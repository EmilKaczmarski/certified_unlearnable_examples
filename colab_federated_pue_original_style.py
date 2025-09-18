#!/usr/bin/env python3
"""
Federated Learning with PUE Noise - Original Codebase Style
Aligned with the original repository's noise generation and recovery methods
Uses pre-trained clean model and implements patch-based PUE noise
"""

# ============================================================================
# PART 1: Imports and Setup
# ============================================================================

import os
import sys
import json
import copy
import time
import random
import pickle
import collections
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# PART 2: Configuration
# ============================================================================

class Config:
    """Central configuration for the experiment"""
    # Paths
    DRIVE_PATH = '/content/drive/MyDrive/diploma_wip'
    DATA_PATH = './data'

    # Pre-trained model path
    PRETRAINED_MODEL_PATH = '/content/drive/MyDrive/diploma_wip/exp_20250916_021135/models/clean_model.pth'

    # Federated Learning
    NUM_CLIENTS = 4
    NUM_ROUNDS = 30
    LOCAL_EPOCHS = 5
    FRACTION_FIT = 1.0  # All clients participate

    # Model and Training
    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # PUE Noise Parameters (Original style)
    EPSILON = 8.0  # Standard perturbation budget
    NUM_STEPS = 60  # Increased iterations
    STEP_SIZE = 2.0
    ATTACK_ITERATIONS = 60  # More iterations for better convergence
    ATTACK_TYPE = 'min-min'
    UNIVERSAL_STOP_ERROR = 0.1

    # Patch-based noise parameters
    PATCH_SIZE = 10  # 10x10 patch size like original
    PATCH_LOCATION = 'center'  # 'center' or 'random'
    PERTURB_TYPE = 'classwise'

    # Recovery Parameters
    RECOVERY_RATE = 0.2
    ETA_VALUES = [0.4, 0.8]
    RECOVERY_EPOCHS = 20
    RECOVERY_LR = 0.01
    GRAD_CLIP = 5.0  # Gradient clipping like original

    # Data Distribution
    USE_IID = True  # Set to True for random split, False for non-IID
    NON_IID_ALPHA = 0.5  # Only used if USE_IID = False

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PART 3: Utility Functions from Original Codebase
# ============================================================================

def rand_bbox(size, patch_size):
    """Generate random bounding box for patch placement"""
    W = size[2]
    H = size[1]

    if Config.PATCH_LOCATION == 'center':
        # Center placement
        cut_x = W // 2 - patch_size // 2
        cut_y = H // 2 - patch_size // 2
    else:
        # Random placement
        cut_x = np.random.randint(0, W - patch_size)
        cut_y = np.random.randint(0, H - patch_size)

    bbx1 = cut_x
    bby1 = cut_y
    bbx2 = cut_x + patch_size
    bby2 = cut_y + patch_size

    return bbx1, bby1, bbx2, bby2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def project_parameters(model, poison_model, origin_params, eta):
    """Project parameters to L2 ball around origin (original style)"""
    # Compute L2 difference
    l2_diff = 0
    for name, param in model.named_parameters():
        diff = param.data - origin_params[name]
        l2_diff += torch.norm(diff) ** 2
    l2_diff = torch.sqrt(l2_diff)

    # Project if needed
    if l2_diff > eta:
        for name, param in model.named_parameters():
            upsilon = (param.data - origin_params[name]) * eta / l2_diff
            param.data = origin_params[name] + upsilon

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ============================================================================
# PART 4: Setup Functions
# ============================================================================

def setup_environment():
    """Setup environment and directories"""
    # Try to mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
        Config.DRIVE_PATH = './experiment_results'

    # Create experiment directory
    os.makedirs(Config.DRIVE_PATH, exist_ok=True)
    exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(Config.DRIVE_PATH, f"exp_original_{exp_timestamp}")

    # Create subdirectories
    for subdir in ['models', 'noise', 'logs', 'plots']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir, IN_COLAB

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# PART 5: Model Definition
# ============================================================================

def create_model():
    """Create ResNet18 model for CIFAR-10"""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, Config.NUM_CLASSES)

    # Adjust for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model

def load_pretrained_model():
    """Load pre-trained clean model"""
    model = create_model()

    try:
        # Try to load the state dict
        state_dict = torch.load(Config.PRETRAINED_MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ Loaded pre-trained model from {Config.PRETRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading pre-trained model: {e}")
        print("🔄 Training new model instead...")
        return None

    return model

# ============================================================================
# PART 6: Data Preparation
# ============================================================================

def create_iid_splits(train_dataset, num_clients: int):
    """Create IID (random) data splits"""
    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    # Split indices evenly among clients
    samples_per_client = num_samples // num_clients
    client_indices = []

    for i in range(num_clients):
        start = i * samples_per_client
        if i == num_clients - 1:  # Last client gets remaining samples
            end = num_samples
        else:
            end = (i + 1) * samples_per_client
        client_indices.append(indices[start:end])

    return client_indices

def create_non_iid_splits(train_dataset, num_clients: int, alpha: float = 0.5):
    """Create non-IID data splits using Dirichlet distribution"""
    num_samples = len(train_dataset)
    labels = np.array([train_dataset[i][1] for i in range(num_samples)])
    num_classes = len(np.unique(labels))

    # Create label indices
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Dirichlet distribution for non-IID split
    client_indices = [[] for _ in range(num_clients)]

    for class_idx in range(num_classes):
        class_size = len(label_indices[class_idx])
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * class_size).astype(int)
        proportions[-1] = class_size - proportions[:-1].sum()

        idx_list = label_indices[class_idx]
        np.random.shuffle(idx_list)

        start = 0
        for client_id, proportion in enumerate(proportions):
            client_indices[client_id].extend(idx_list[start:start + proportion])
            start += proportion

    # Shuffle indices for each client
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices

def prepare_data():
    """Prepare CIFAR-10 dataset with transforms"""
    # Repository-style transforms (no normalization)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH, train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_PATH, train=False, download=True, transform=transform_test
    )

    # Create data splits based on configuration
    if Config.USE_IID:
        print("Using IID (random) data distribution")
        client_indices = create_iid_splits(train_dataset, Config.NUM_CLIENTS)
    else:
        print(f"Using non-IID data distribution (α={Config.NON_IID_ALPHA})")
        client_indices = create_non_iid_splits(train_dataset, Config.NUM_CLIENTS, Config.NON_IID_ALPHA)

    # Create client datasets
    client_datasets = []
    for indices in client_indices:
        client_datasets.append(Subset(train_dataset, indices))

    return client_datasets, test_dataset

# ============================================================================
# PART 7: Original Style PUE Noise Generator
# ============================================================================

class OriginalStylePUEGenerator:
    """PUE noise generator following original codebase patterns"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _patch_noise_extend_to_img(self, noise, image_size, patch_location='center'):
        """Extend patch noise to full image size (original style)"""
        c, h, w = image_size
        mask = np.zeros((c, h, w), dtype=np.float32)

        # Get patch coordinates
        bbx1, bby1, bbx2, bby2 = rand_bbox((1, h, w), Config.PATCH_SIZE)

        # Place noise patch
        mask[:, bbx1:bbx2, bby1:bby2] = noise.cpu().numpy()

        return (bbx1, bbx2, bby1, bby2), torch.from_numpy(mask).to(self.device)

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None):
        """Min-min attack similar to original implementation"""
        model.eval()

        # Initialize perturbation
        eta = torch.zeros_like(images).to(self.device)
        if random_noise is not None:
            eta = random_noise.clone()

        eta.requires_grad = True

        # Iterative attack
        for step in range(Config.NUM_STEPS):
            if eta.grad is not None:
                eta.grad.zero_()

            perturb_img = images + eta
            perturb_img = torch.clamp(perturb_img, 0, 1)

            outputs = model(perturb_img)
            loss = criterion(outputs, labels)

            # Min-min: minimize loss to make examples unlearnable
            loss.backward()

            # Update perturbation
            eta_grad = eta.grad.data
            eta.data = eta.data - Config.STEP_SIZE/255 * torch.sign(eta_grad)

            # Project to epsilon ball
            eta.data = torch.clamp(eta.data, -Config.EPSILON/255, Config.EPSILON/255)

            # Ensure valid image range
            eta.data = torch.clamp(images + eta.data, 0, 1) - images

        final_perturb = images + eta.data
        return final_perturb, eta.data

    def generate_classwise_noise_original_style(self, data_loader, exp_dir):
        """Generate class-wise PUE noise following original patterns"""
        print("\n" + "="*50)
        print(f"Generating Original-Style PUE Noise")
        print(f"Epsilon: {Config.EPSILON}, Patch Size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
        print("="*50)

        self.model.eval()
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        # Initialize class-wise patch noise
        class_noise = {}
        for i in range(Config.NUM_CLASSES):
            # Patch-sized noise instead of full image
            class_noise[i] = torch.zeros(3, Config.PATCH_SIZE, Config.PATCH_SIZE).uniform_(
                -Config.EPSILON/255, Config.EPSILON/255
            ).to(self.device)

        # Optimization loop with proper convergence checking
        condition = True
        iteration = 0

        while condition and iteration < Config.ATTACK_ITERATIONS:
            class_updates = collections.defaultdict(list)

            # Process batches
            batch_count = 0
            for images, labels in tqdm(data_loader, desc=f"Attack Iteration {iteration+1}"):
                if batch_count > 30:  # More batches for better convergence
                    break
                batch_count += 1

                images, labels = images.to(self.device), labels.to(self.device)

                # Apply patch noise to images
                batch_noise = []
                mask_coord_list = []

                for i, label in enumerate(labels):
                    label_item = label.item()
                    if label_item in class_noise:
                        mask_coord, class_noise_extended = self._patch_noise_extend_to_img(
                            class_noise[label_item], images[i].shape
                        )
                        batch_noise.append(class_noise_extended)
                        mask_coord_list.append(mask_coord)
                    else:
                        # Fallback
                        batch_noise.append(torch.zeros_like(images[i]))
                        mask_coord_list.append((0, Config.PATCH_SIZE, 0, Config.PATCH_SIZE))

                # Stack and apply noise
                batch_noise = torch.stack(batch_noise).to(self.device)

                # Perform min-min attack
                perturb_img, eta = self.min_min_attack(
                    images, labels, self.model, None, criterion, random_noise=batch_noise
                )

                # Extract patch updates
                class_noise_eta = collections.defaultdict(list)
                for i in range(len(eta)):
                    x1, x2, y1, y2 = mask_coord_list[i]
                    delta = eta[i][:, x1:x2, y1:y2]
                    class_noise_eta[labels[i].item()].append(delta.detach().cpu())

                # Update class noise
                for key in class_noise_eta:
                    if class_noise_eta[key]:  # Check if list is not empty
                        delta_stack = torch.stack(class_noise_eta[key])
                        delta_mean = delta_stack.mean(dim=0) - class_noise[key].cpu()
                        class_noise_update = class_noise[key].cpu() + delta_mean
                        class_noise[key] = torch.clamp(
                            class_noise_update,
                            -Config.EPSILON/255,
                            Config.EPSILON/255
                        ).to(self.device)

            # Evaluate termination condition
            if (iteration + 1) % 10 == 0:
                error_rate = self.universal_perturbation_eval(data_loader, class_noise)
                print(f"Iteration {iteration+1}: Error rate: {error_rate:.3f}")

                # Check convergence for min-min attack
                condition = error_rate > Config.UNIVERSAL_STOP_ERROR
                if error_rate < 0.5:  # If accuracy > 50%, continue
                    condition = True

            iteration += 1

        # Save noise
        noise_path = os.path.join(exp_dir, 'noise', 'original_style_pue_noise.pth')
        torch.save(class_noise, noise_path)
        print(f"Original-style PUE noise saved to {noise_path}")

        return class_noise

    def universal_perturbation_eval(self, data_loader, class_noise):
        """Evaluate universal perturbation effectiveness"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx > 15:  # Quick evaluation
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                # Apply noise
                for i in range(len(labels)):
                    label_item = labels[i].item()
                    if label_item in class_noise:
                        mask_coord, class_noise_extended = self._patch_noise_extend_to_img(
                            class_noise[label_item], images[i].shape
                        )
                        images[i] += class_noise_extended
                        images[i] = torch.clamp(images[i], 0, 1)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        error_rate = 1 - (correct / total) if total > 0 else 1
        return error_rate

# ============================================================================
# PART 8: Original Style Recovery
# ============================================================================

class OriginalStyleRecovery:
    """Recovery experiment following original trainer patterns"""

    def __init__(self, poisoned_model, clean_data_loader, test_loader, device, model_type='poisoned'):
        self.poisoned_model = copy.deepcopy(poisoned_model)
        self.clean_data_loader = clean_data_loader
        self.test_loader = test_loader
        self.device = device
        self.model_type = model_type

        # Meters for tracking (original style)
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()

    def train_recovery_batch_original_style(self, images, labels, model, poison_model, origin_param, eta, optimizer):
        """Single batch recovery training (original style)"""
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping (original style)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        # Project parameters (original style - after each step)
        project_parameters(model, poison_model, origin_param, eta)

        # Calculate accuracy
        acc, = accuracy(logits, labels, topk=(1,))
        acc = acc.item()

        # Update meters
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])

        return {
            "acc": acc,
            "acc_avg": self.acc_meters.avg,
            "loss": loss.item(),
            "loss_avg": self.loss_meters.avg,
            "lr": optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm.item()
        }

    def train_recovery_original_style(self, eta_value, exp_dir):
        """Recovery training following original patterns"""
        print(f"\n{'='*50}")
        print(f"Original Style Recovery: {self.model_type} model, η={eta_value}")
        print(f"{'='*50}")

        # Create recovery model
        recovery_model = copy.deepcopy(self.poisoned_model).to(self.device)

        # Store origin parameters (original style)
        origin_param = {}
        for name, param in self.poisoned_model.named_parameters():
            origin_param[name] = param.data.clone()

        # Setup optimizer and scheduler (original style)
        optimizer = optim.SGD(
            recovery_model.parameters(),
            lr=Config.RECOVERY_LR,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.RECOVERY_EPOCHS,
            eta_min=0.0
        )

        history = []

        # Training loop
        for epoch in tqdm(range(Config.RECOVERY_EPOCHS), desc=f"Recovery (η={eta_value})"):
            # Reset meters
            self.loss_meters.reset()
            self.acc_meters.reset()

            recovery_model.train()

            # Batch training (original style)
            for images, labels in self.clean_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Train single batch with original style
                log_payload = self.train_recovery_batch_original_style(
                    images, labels, recovery_model,
                    self.poisoned_model, origin_param, eta_value, optimizer
                )

            # Step scheduler
            scheduler.step()

            # Evaluate on test set
            test_loss, test_acc = self.evaluate_model(recovery_model)

            # Compute L2 difference
            l2_diff = self.compute_l2_diff(recovery_model)

            # Record history
            history.append({
                'epoch': epoch + 1,
                'test_accuracy': test_acc,
                'train_accuracy': self.acc_meters.avg,
                'train_loss': self.loss_meters.avg,
                'test_loss': test_loss,
                'l2_diff': l2_diff,
                'lr': optimizer.param_groups[0]['lr']
            })

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}: Train Acc: {self.acc_meters.avg:.2f}%, "
                      f"Test Acc: {test_acc:.2f}%, L2 Diff: {l2_diff:.4f}")

        # Save results
        results_path = os.path.join(exp_dir, 'logs', f'original_recovery_{self.model_type}_eta_{eta_value}.json')
        with open(results_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save recovered model
        model_path = os.path.join(exp_dir, 'models', f'original_recovered_{self.model_type}_eta_{eta_value}.pth')
        torch.save(recovery_model.state_dict(), model_path)

        return history

    def compute_l2_diff(self, model):
        """Compute L2 difference from poisoned model"""
        diff = 0
        for param1, param2 in zip(model.parameters(), self.poisoned_model.parameters()):
            diff += torch.norm(param1 - param2) ** 2
        return torch.sqrt(diff).item()

    def evaluate_model(self, model):
        """Evaluate model on test set"""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

# ============================================================================
# PART 9: Federated Learning Functions
# ============================================================================

def train_client_with_patch_noise(model, data_loader, epochs, device, class_noise):
    """Train client with patch-based PUE noise"""
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )

    total_loss = 0
    correct = 0
    total = 0

    # Create noise generator for applying patches
    noise_gen = OriginalStylePUEGenerator(model, device)

    for epoch in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Apply patch noise
            noisy_images = images.clone()
            for i in range(len(labels)):
                label = labels[i].item()
                if label in class_noise:
                    mask_coord, class_noise_extended = noise_gen._patch_noise_extend_to_img(
                        class_noise[label], images[i].shape
                    )
                    noisy_images[i] += class_noise_extended
                    noisy_images[i] = torch.clamp(noisy_images[i], 0, 1)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / (len(data_loader) * epochs)
    accuracy = correct / total

    return avg_loss, accuracy

def federated_averaging(global_model, client_models, client_sizes):
    """Perform federated averaging"""
    total_size = sum(client_sizes)

    # Initialize aggregated state dict
    aggregated_state = {}

    # Get state dicts from all clients
    client_states = [model.state_dict() for model in client_models]

    # Aggregate parameters
    for key in client_states[0].keys():
        # Initialize with zeros
        aggregated_state[key] = torch.zeros_like(client_states[0][key])

        # Weighted average
        for i, state_dict in enumerate(client_states):
            weight = client_sizes[i] / total_size
            if aggregated_state[key].dtype in [torch.long, torch.int64, torch.int32]:
                # For integer tensors, take the first client's value
                if i == 0:
                    aggregated_state[key] = state_dict[key].clone()
            else:
                # For float tensors, perform weighted average
                aggregated_state[key] += weight * state_dict[key]

    # Update global model
    global_model.load_state_dict(aggregated_state)
    return global_model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def run_federated_learning_with_patch_noise(client_datasets, test_loader, exp_dir, class_noise):
    """Run federated learning with patch-based noise"""
    device = Config.DEVICE

    # Initialize global model
    global_model = create_model().to(device)

    # Track history
    history = {'rounds': [], 'train_acc': [], 'test_acc': []}

    print(f"\n{'='*50}")
    print(f"Federated Learning with Original-Style PUE Noise")
    print(f"{'='*50}")

    # Training rounds
    for round_num in tqdm(range(Config.NUM_ROUNDS), desc="FL Rounds"):
        round_train_accs = []
        client_models = []
        client_sizes = []

        # Train each client
        for client_id in range(Config.NUM_CLIENTS):
            # Create client model and copy global weights
            client_model = create_model().to(device)
            client_model.load_state_dict(global_model.state_dict())

            # Create data loader for client
            client_loader = DataLoader(
                client_datasets[client_id],
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )

            # Train client with patch noise
            loss, acc = train_client_with_patch_noise(
                client_model, client_loader,
                Config.LOCAL_EPOCHS, device, class_noise
            )

            round_train_accs.append(acc)
            client_models.append(client_model)
            client_sizes.append(len(client_datasets[client_id]))

        # Aggregate client updates
        global_model = federated_averaging(global_model, client_models, client_sizes)

        # Evaluate global model periodically
        if (round_num + 1) % 5 == 0 or round_num == Config.NUM_ROUNDS - 1:
            test_loss, test_acc = evaluate_model(global_model, test_loader, device)
            avg_train_acc = np.mean(round_train_accs)

            history['rounds'].append(round_num + 1)
            history['train_acc'].append(avg_train_acc * 100)
            history['test_acc'].append(test_acc)

            if (round_num + 1) % 10 == 0:
                print(f"\nRound {round_num + 1}: Train Acc: {avg_train_acc*100:.2f}%, Test Acc: {test_acc:.2f}%")

    return global_model, history

# ============================================================================
# PART 10: Main Experiment
# ============================================================================

def main():
    """Main experiment pipeline with original-style implementation"""
    print("="*60)
    print("FEDERATED LEARNING WITH ORIGINAL-STYLE PUE NOISE")
    print("="*60)

    # Setup
    exp_dir, in_colab = setup_environment()
    set_seed(42)
    device = Config.DEVICE

    print(f"Device: {device}")
    print(f"Experiment directory: {exp_dir}")

    # Prepare data
    print("\nPreparing datasets...")
    client_datasets, test_dataset = prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Log data distribution
    for i, dataset in enumerate(client_datasets):
        labels = [dataset[j][1] for j in range(len(dataset))]
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Client {i}: {len(dataset)} samples, distribution: {dist}")

    # ============================================================================
    # STAGE 1: Load Pre-trained Clean Model
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 1: Loading Pre-trained Clean Model")
    print("="*50)

    clean_model = load_pretrained_model()
    if clean_model is None:
        print("❌ Could not load pre-trained model. Please check the path.")
        return

    clean_model = clean_model.to(device)

    # Evaluate pre-trained model
    clean_loss, clean_acc = evaluate_model(clean_model, test_loader, device)
    print(f"Pre-trained Clean Model Accuracy: {clean_acc:.2f}%")

    # ============================================================================
    # STAGE 2: Generate Original-Style PUE Noise
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 2: Generating Original-Style PUE Noise")
    print("="*50)

    noise_gen_loader = DataLoader(
        client_datasets[0],  # Use first client's data
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    noise_generator = OriginalStylePUEGenerator(clean_model, device)
    class_noise = noise_generator.generate_classwise_noise_original_style(noise_gen_loader, exp_dir)

    # ============================================================================
    # STAGE 3: Train on PUE-Poisoned Data with Original Style
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 3: Training on Original-Style PUE-Poisoned Data")
    print("="*50)

    poisoned_model, poisoned_history = run_federated_learning_with_patch_noise(
        client_datasets, test_loader, exp_dir, class_noise
    )

    # Save poisoned model
    poisoned_path = os.path.join(exp_dir, 'models', 'original_poisoned_model.pth')
    torch.save(poisoned_model.state_dict(), poisoned_path)

    # Final evaluation
    poisoned_loss, poisoned_acc = evaluate_model(poisoned_model, test_loader, device)
    print(f"Original-Style Poisoned Model Final Accuracy: {poisoned_acc:.2f}%")

    # ============================================================================
    # STAGE 4: Original-Style Recovery Experiments
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 4: Original-Style Recovery Experiments")
    print("="*50)

    # Prepare recovery data
    recovery_size = int(Config.RECOVERY_RATE * len(client_datasets[0]))
    recovery_indices = random.sample(range(len(client_datasets[0])), recovery_size)
    recovery_dataset = Subset(client_datasets[0].dataset,
                             [client_datasets[0].indices[i] for i in recovery_indices])
    recovery_loader = DataLoader(recovery_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Using {len(recovery_dataset)} samples for recovery")

    # Store all recovery results
    all_recovery_results = {'poisoned': {}, 'clean': {}}

    # Recovery on poisoned model
    for eta in Config.ETA_VALUES:
        recovery_exp = OriginalStyleRecovery(
            poisoned_model, recovery_loader, test_loader, device, 'poisoned'
        )
        history = recovery_exp.train_recovery_original_style(eta, exp_dir)
        all_recovery_results['poisoned'][eta] = history

    # Recovery on clean model (baseline)
    for eta in Config.ETA_VALUES:
        recovery_exp = OriginalStyleRecovery(
            clean_model, recovery_loader, test_loader, device, 'clean'
        )
        history = recovery_exp.train_recovery_original_style(eta, exp_dir)
        all_recovery_results['clean'][eta] = history

    # ============================================================================
    # STAGE 5: Generate Plots and Summary
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 5: Generating Results")
    print("="*50)

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Federated Learning Progress (compare with clean baseline)
    ax = axes[0, 0]
    ax.plot(poisoned_history['rounds'], poisoned_history['test_acc'], 'r-', label='Poisoned (Original Style)')
    ax.axhline(y=clean_acc, color='b', linestyle='--', label='Clean Baseline')
    ax.set_xlabel('Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Federated Learning Progress')
    ax.legend()
    ax.grid(True)

    # Plot 2: Recovery - Poisoned Model
    ax = axes[0, 1]
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['poisoned'][eta]]
        accs = [h['test_accuracy'] for h in all_recovery_results['poisoned'][eta]]
        ax.plot(epochs, accs, label=f'η={eta}')
    ax.axhline(y=clean_acc, color='b', linestyle='--', alpha=0.5, label='Clean')
    ax.axhline(y=poisoned_acc, color='r', linestyle='--', alpha=0.5, label='Poisoned')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Recovery from Poisoned Model')
    ax.legend()
    ax.grid(True)

    # Plot 3: Recovery - Clean Model
    ax = axes[0, 2]
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['clean'][eta]]
        accs = [h['test_accuracy'] for h in all_recovery_results['clean'][eta]]
        ax.plot(epochs, accs, label=f'η={eta}')
    ax.axhline(y=clean_acc, color='b', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Recovery from Clean Model')
    ax.legend()
    ax.grid(True)

    # Plot 4: Training Progress During Recovery (Poisoned)
    ax = axes[1, 0]
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['poisoned'][eta]]
        train_accs = [h['train_accuracy'] for h in all_recovery_results['poisoned'][eta]]
        ax.plot(epochs, train_accs, label=f'η={eta}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Progress (Poisoned Recovery)')
    ax.legend()
    ax.grid(True)

    # Plot 5: L2 Distance Evolution
    ax = axes[1, 1]
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['poisoned'][eta]]
        l2s = [h['l2_diff'] for h in all_recovery_results['poisoned'][eta]]
        ax.plot(epochs, l2s, label=f'Poisoned η={eta}')
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['clean'][eta]]
        l2s = [h['l2_diff'] for h in all_recovery_results['clean'][eta]]
        ax.plot(epochs, l2s, '--', label=f'Clean η={eta}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2 Distance')
    ax.set_title('L2 Distance Evolution')
    ax.legend()
    ax.grid(True)

    # Plot 6: Final Comparison
    ax = axes[1, 2]
    eta_labels = [str(eta) for eta in Config.ETA_VALUES]
    poisoned_finals = [all_recovery_results['poisoned'][eta][-1]['test_accuracy'] for eta in Config.ETA_VALUES]
    clean_finals = [all_recovery_results['clean'][eta][-1]['test_accuracy'] for eta in Config.ETA_VALUES]

    x = np.arange(len(eta_labels))
    width = 0.35
    ax.bar(x - width/2, poisoned_finals, width, label='Poisoned Recovery', color='orange')
    ax.bar(x + width/2, clean_finals, width, label='Clean Recovery', color='green')
    ax.axhline(y=clean_acc, color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=poisoned_acc, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Eta Value')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Final Recovery Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(eta_labels)
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, 'plots', 'original_style_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # Generate summary
    summary = {
        'config': {
            'epsilon': Config.EPSILON,
            'num_steps': Config.NUM_STEPS,
            'patch_size': Config.PATCH_SIZE,
            'attack_iterations': Config.ATTACK_ITERATIONS,
            'num_clients': Config.NUM_CLIENTS,
            'num_rounds': Config.NUM_ROUNDS,
            'eta_values': Config.ETA_VALUES,
            'implementation_style': 'original_codebase_style'
        },
        'results': {
            'clean_accuracy': clean_acc,
            'poisoned_accuracy': poisoned_acc,
            'accuracy_drop': clean_acc - poisoned_acc,
            'recovery': {}
        }
    }

    for eta in Config.ETA_VALUES:
        summary['results']['recovery'][f'eta_{eta}'] = {
            'poisoned_final': all_recovery_results['poisoned'][eta][-1]['test_accuracy'],
            'clean_final': all_recovery_results['clean'][eta][-1]['test_accuracy'],
            'improvement': all_recovery_results['poisoned'][eta][-1]['test_accuracy'] - poisoned_acc
        }

    summary_path = os.path.join(exp_dir, 'logs', 'original_style_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "="*60)
    print("ORIGINAL-STYLE EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Epsilon: {Config.EPSILON}")
    print(f"  Patch Size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"  Attack Iterations: {Config.ATTACK_ITERATIONS}")

    print(f"\nBaseline Accuracies:")
    print(f"  Clean Model: {clean_acc:.2f}%")
    print(f"  Poisoned Model: {poisoned_acc:.2f}%")
    print(f"  Accuracy Drop: {clean_acc - poisoned_acc:.2f}%")

    print(f"\nRecovery Results:")
    for eta in Config.ETA_VALUES:
        print(f"\n  η = {eta}:")
        print(f"    Poisoned Recovery: {all_recovery_results['poisoned'][eta][-1]['test_accuracy']:.2f}%")
        print(f"    Clean Recovery: {all_recovery_results['clean'][eta][-1]['test_accuracy']:.2f}%")
        print(f"    Improvement: {all_recovery_results['poisoned'][eta][-1]['test_accuracy'] - poisoned_acc:.2f}%")

    print(f"\nResults saved to: {exp_dir}")

    return summary

if __name__ == "__main__":
    results = main()
#!/usr/bin/env python3
"""
Simplified Federated Learning with PUE Noise on CIFAR-10
Direct implementation without Flower simulation complexity
Includes multiple eta recovery experiments
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

    # PUE Noise Parameters
    EPSILON = 32.0  # Large perturbation
    NUM_STEPS = 20
    STEP_SIZE = 2.0
    ATTACK_ITERATIONS = 30

    # Recovery Parameters
    RECOVERY_RATE = 0.2
    ETA_VALUES = [0.4, 0.8]
    RECOVERY_EPOCHS = 20
    RECOVERY_LR = 0.01

    # Data Distribution
    USE_IID = True  # Set to True for random split, False for non-IID
    NON_IID_ALPHA = 0.5  # Only used if USE_IID = False

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PART 3: Setup Functions
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
    exp_dir = os.path.join(Config.DRIVE_PATH, f"exp_{exp_timestamp}")

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
# PART 4: Model Definition
# ============================================================================

def create_model():
    """Create ResNet18 model for CIFAR-10"""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, Config.NUM_CLASSES)

    # Adjust for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model

# ============================================================================
# PART 5: Data Preparation
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
# PART 6: PUE Noise Generation
# ============================================================================

class PUENoiseGenerator:
    """Generate PUE (Provably Unlearnable Examples) noise"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate_classwise_noise(self, data_loader, exp_dir):
        """Generate class-wise PUE noise with large perturbations"""
        print("\n" + "="*50)
        print(f"Generating PUE Noise (ε={Config.EPSILON})")
        print("="*50)

        self.model.eval()
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        # Initialize noise for each class
        class_noise = {}
        for i in range(Config.NUM_CLASSES):
            class_noise[i] = torch.zeros(3, 32, 32).uniform_(
                -Config.EPSILON/255, Config.EPSILON/255
            ).to(self.device)

        # Optimization loop
        for iteration in range(Config.ATTACK_ITERATIONS):
            class_updates = {i: [] for i in range(Config.NUM_CLASSES)}

            # Process batches
            batch_count = 0
            for images, labels in tqdm(data_loader, desc=f"Iteration {iteration+1}/{Config.ATTACK_ITERATIONS}"):
                if batch_count > 20:  # Limit batches for speed
                    break
                batch_count += 1

                images, labels = images.to(self.device), labels.to(self.device)

                # Add current noise to images
                noisy_images = images.clone()
                for i in range(len(labels)):
                    label = labels[i].item()
                    noisy_images[i] += class_noise[label]
                    noisy_images[i] = torch.clamp(noisy_images[i], 0, 1)

                # Compute gradients
                noisy_images.requires_grad = True
                outputs = self.model(noisy_images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Collect gradients per class
                for i in range(len(labels)):
                    label = labels[i].item()
                    grad = noisy_images.grad[i].detach()
                    class_updates[label].append(grad)

            # Update class noise
            for class_idx in range(Config.NUM_CLASSES):
                if class_updates[class_idx]:
                    avg_grad = torch.stack(class_updates[class_idx]).mean(dim=0)
                    # Min-min attack: minimize loss
                    update = (Config.STEP_SIZE/255) * torch.sign(avg_grad)
                    class_noise[class_idx] -= update
                    class_noise[class_idx] = torch.clamp(
                        class_noise[class_idx],
                        -Config.EPSILON/255,
                        Config.EPSILON/255
                    )

            # Evaluate effectiveness every 10 iterations
            if (iteration + 1) % 10 == 0:
                acc = self.evaluate_noise(data_loader, class_noise)
                print(f"Iteration {iteration+1}: Accuracy on noisy data: {acc:.2f}%")
                if acc < 40:  # Target reached
                    print("Target accuracy reached!")
                    break

        # Save noise
        noise_path = os.path.join(exp_dir, 'noise', 'pue_noise.pth')
        torch.save(class_noise, noise_path)
        print(f"PUE noise saved to {noise_path}")

        return class_noise

    def evaluate_noise(self, data_loader, class_noise):
        """Quick evaluation of noise effectiveness"""
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx > 10:  # Quick evaluation
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                # Add noise
                for i in range(len(labels)):
                    label = labels[i].item()
                    images[i] += class_noise[label]
                    images[i] = torch.clamp(images[i], 0, 1)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total if total > 0 else 0

# ============================================================================
# PART 7: Federated Learning Functions
# ============================================================================

def train_client(model, data_loader, epochs, device):
    """Train a client model"""
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

    for epoch in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
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

def train_client_with_noise(model, data_loader, epochs, device, class_noise):
    """Train a client model with PUE noise"""
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

    for epoch in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Add PUE noise
            noisy_images = images.clone()
            for i in range(len(labels)):
                label = labels[i].item()
                if label in class_noise:
                    noisy_images[i] += class_noise[label].to(device)
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

def run_federated_learning(client_datasets, test_loader, exp_dir, use_noise=False, class_noise=None):
    """Run federated learning simulation"""
    device = Config.DEVICE

    # Initialize global model
    global_model = create_model().to(device)

    # Track history
    history = {'rounds': [], 'train_acc': [], 'test_acc': []}

    print(f"\n{'='*50}")
    print(f"Federated Learning ({'with PUE noise' if use_noise else 'clean data'})")
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

            # Train client
            if use_noise and class_noise is not None:
                loss, acc = train_client_with_noise(
                    client_model, client_loader,
                    Config.LOCAL_EPOCHS, device, class_noise
                )
            else:
                loss, acc = train_client(
                    client_model, client_loader,
                    Config.LOCAL_EPOCHS, device
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
# PART 8: Recovery Experiments
# ============================================================================

def recovery_experiment(poisoned_model, clean_data_loader, test_loader, eta_value, exp_dir, model_type='poisoned'):
    """Run recovery experiment with projected gradient descent"""
    device = Config.DEVICE

    print(f"\n{'='*50}")
    print(f"Recovery: {model_type} model, η={eta_value}")
    print(f"{'='*50}")

    # Create recovery model
    recovery_model = copy.deepcopy(poisoned_model).to(device)
    origin_params = {name: param.data.clone() for name, param in poisoned_model.named_parameters()}

    optimizer = optim.SGD(recovery_model.parameters(), lr=Config.RECOVERY_LR, momentum=Config.MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in tqdm(range(Config.RECOVERY_EPOCHS), desc=f"Recovery (η={eta_value})"):
        # Train on clean data
        recovery_model.train()
        for images, labels in clean_data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = recovery_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Project parameters
            with torch.no_grad():
                # Compute L2 distance
                l2_diff = 0
                for name, param in recovery_model.named_parameters():
                    diff = param.data - origin_params[name]
                    l2_diff += torch.norm(diff) ** 2
                l2_diff = torch.sqrt(l2_diff)

                # Project if needed
                if l2_diff > eta_value:
                    for name, param in recovery_model.named_parameters():
                        direction = param.data - origin_params[name]
                        param.data = origin_params[name] + direction * (eta_value / l2_diff)

        # Evaluate
        test_loss, test_acc = evaluate_model(recovery_model, test_loader, device)
        history.append({
            'epoch': epoch + 1,
            'test_accuracy': test_acc,
            'l2_diff': l2_diff.item()
        })

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Test Acc: {test_acc:.2f}%, L2 Diff: {l2_diff.item():.4f}")

    # Save results
    results_path = os.path.join(exp_dir, 'logs', f'recovery_{model_type}_eta_{eta_value}.json')
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)

    return history

# ============================================================================
# PART 9: Main Experiment
# ============================================================================

def main():
    """Main experiment pipeline"""
    print("="*60)
    print("FEDERATED LEARNING WITH PUE NOISE EXPERIMENT")
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
    # STAGE 1: Train Clean Federated Model
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 1: Training Clean Federated Model")
    print("="*50)

    clean_model, clean_history = run_federated_learning(
        client_datasets, test_loader, exp_dir, use_noise=False
    )

    # Save clean model
    clean_path = os.path.join(exp_dir, 'models', 'clean_model.pth')
    torch.save(clean_model.state_dict(), clean_path)

    # Final evaluation
    clean_loss, clean_acc = evaluate_model(clean_model, test_loader, device)
    print(f"Clean Model Final Accuracy: {clean_acc:.2f}%")

    # ============================================================================
    # STAGE 2: Generate PUE Noise
    # ============================================================================

    noise_gen_loader = DataLoader(
        client_datasets[0],  # Use first client's data
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    noise_generator = PUENoiseGenerator(clean_model, device)
    class_noise = noise_generator.generate_classwise_noise(noise_gen_loader, exp_dir)

    # ============================================================================
    # STAGE 3: Train on PUE-Poisoned Data
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 3: Training on PUE-Poisoned Data")
    print("="*50)

    poisoned_model, poisoned_history = run_federated_learning(
        client_datasets, test_loader, exp_dir, use_noise=True, class_noise=class_noise
    )

    # Save poisoned model
    poisoned_path = os.path.join(exp_dir, 'models', 'poisoned_model.pth')
    torch.save(poisoned_model.state_dict(), poisoned_path)

    # Final evaluation
    poisoned_loss, poisoned_acc = evaluate_model(poisoned_model, test_loader, device)
    print(f"Poisoned Model Final Accuracy: {poisoned_acc:.2f}%")

    # ============================================================================
    # STAGE 4: Recovery Experiments
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 4: Recovery Experiments")
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
        history = recovery_experiment(
            poisoned_model, recovery_loader, test_loader, eta, exp_dir, 'poisoned'
        )
        all_recovery_results['poisoned'][eta] = history

    # Recovery on clean model (baseline)
    for eta in Config.ETA_VALUES:
        history = recovery_experiment(
            clean_model, recovery_loader, test_loader, eta, exp_dir, 'clean'
        )
        all_recovery_results['clean'][eta] = history

    # ============================================================================
    # STAGE 5: Generate Plots and Summary
    # ============================================================================

    print("\n" + "="*50)
    print("STAGE 5: Generating Results")
    print("="*50)

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Federated Learning Progress
    ax = axes[0, 0]
    ax.plot(clean_history['rounds'], clean_history['test_acc'], 'b-', label='Clean')
    ax.plot(poisoned_history['rounds'], poisoned_history['test_acc'], 'r-', label='Poisoned')
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
    ax.set_title('Recovery from Clean Model (Baseline)')
    ax.legend()
    ax.grid(True)

    # Plot 4: L2 Distance - Poisoned
    ax = axes[1, 0]
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['poisoned'][eta]]
        l2s = [h['l2_diff'] for h in all_recovery_results['poisoned'][eta]]
        ax.plot(epochs, l2s, label=f'η={eta}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2 Distance')
    ax.set_title('L2 Distance (Poisoned Model)')
    ax.legend()
    ax.grid(True)

    # Plot 5: L2 Distance - Clean
    ax = axes[1, 1]
    for eta in Config.ETA_VALUES:
        epochs = [h['epoch'] for h in all_recovery_results['clean'][eta]]
        l2s = [h['l2_diff'] for h in all_recovery_results['clean'][eta]]
        ax.plot(epochs, l2s, label=f'η={eta}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2 Distance')
    ax.set_title('L2 Distance (Clean Model)')
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
    plot_path = os.path.join(exp_dir, 'plots', 'results.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # Generate summary
    summary = {
        'config': {
            'num_clients': Config.NUM_CLIENTS,
            'num_rounds': Config.NUM_ROUNDS,
            'epsilon': Config.EPSILON,
            'eta_values': Config.ETA_VALUES,
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

    summary_path = os.path.join(exp_dir, 'logs', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
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
#!/usr/bin/env python3
"""
Recovery Experiment - Eta Parameter Sweep
Uses pre-trained models and PUE noise to run recovery with multiple eta values
Only focuses on recovery experiments with comprehensive eta sweep
"""

# ============================================================================
# PART 1: Imports and Setup
# ============================================================================

import os
import json
import copy
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# PART 2: Configuration
# ============================================================================

class Config:
    """Configuration for recovery experiments"""
    # Paths
    DRIVE_PATH = '/content/drive/MyDrive/diploma_wip'
    DATA_PATH = './data'

    # Pre-trained model and noise paths
    PRETRAINED_CLEAN_MODEL = '/content/drive/MyDrive/diploma_wip/exp_20250916_021135/models/clean_model.pth'
    PRETRAINED_POISONED_MODEL = '/content/drive/MyDrive/diploma_wip/exp_original_20250916_075857/models/original_poisoned_model.pth'
    PUE_NOISE_PATH = '/content/drive/MyDrive/diploma_wip/exp_original_20250916_075857/noise/original_style_pue_noise.pth'

    # Model and Training
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # Recovery Parameters - Extended Eta Sweep
    RECOVERY_RATE = 0.2
    ETA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    RECOVERY_EPOCHS = 20
    RECOVERY_LR = 0.01
    GRAD_CLIP = 5.0

    # Data Configuration
    NUM_CLIENTS = 4
    USE_IID = True

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PART 3: Utility Functions
# ============================================================================

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
    """Project parameters to L2 ball around origin"""
    if eta == 0.0:
        return  # No projection for eta=0

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
# PART 4: Setup Functions
# ============================================================================

def setup_environment():
    """Setup environment and directories"""
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
    exp_dir = os.path.join(Config.DRIVE_PATH, f"recovery_eta_sweep_{exp_timestamp}")

    # Create subdirectories
    for subdir in ['logs', 'plots']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir, IN_COLAB

# ============================================================================
# PART 5: Model Functions
# ============================================================================

def create_model():
    """Create ResNet18 model for CIFAR-10"""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, Config.NUM_CLASSES)

    # Adjust for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model

def load_model(path, device):
    """Load model from path"""
    model = create_model()
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        print(f"✅ Loaded model from {path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {path}: {e}")
        return None

# ============================================================================
# PART 6: Data Preparation
# ============================================================================

def create_iid_splits(train_dataset, num_clients: int):
    """Create IID data splits"""
    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    samples_per_client = num_samples // num_clients
    client_indices = []

    for i in range(num_clients):
        start = i * samples_per_client
        if i == num_clients - 1:
            end = num_samples
        else:
            end = (i + 1) * samples_per_client
        client_indices.append(indices[start:end])

    return client_indices

def prepare_data():
    """Prepare CIFAR-10 dataset"""
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

    # Create IID splits
    print("Using IID (random) data distribution for recovery data")
    client_indices = create_iid_splits(train_dataset, Config.NUM_CLIENTS)

    # Create client datasets
    client_datasets = []
    for indices in client_indices:
        client_datasets.append(Subset(train_dataset, indices))

    return client_datasets, test_dataset

# ============================================================================
# PART 7: Recovery Experiment Class
# ============================================================================

class EtaSweepRecovery:
    """Recovery experiment with eta parameter sweep"""

    def __init__(self, clean_model, poisoned_model, clean_data_loader, test_loader, device):
        self.clean_model = clean_model
        self.poisoned_model = poisoned_model
        self.clean_data_loader = clean_data_loader
        self.test_loader = test_loader
        self.device = device

        # Meters for tracking
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()

    def train_recovery_batch(self, images, labels, model, poison_model, origin_param, eta, optimizer):
        """Single batch recovery training"""
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        # Project parameters (step-wise projection)
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

    def run_recovery_experiment(self, eta_value, exp_dir, model_type='poisoned'):
        """Run recovery experiment for a single eta value"""
        print(f"\n{'='*50}")
        print(f"Recovery Experiment: {model_type} model, η={eta_value}")
        print(f"{'='*50}")

        # Choose base model
        base_model = self.poisoned_model if model_type == 'poisoned' else self.clean_model

        # Create recovery model
        recovery_model = copy.deepcopy(base_model).to(self.device)

        # Store origin parameters
        origin_param = {}
        for name, param in base_model.named_parameters():
            origin_param[name] = param.data.clone()

        # Setup optimizer and scheduler
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
        for epoch in tqdm(range(Config.RECOVERY_EPOCHS), desc=f"Recovery η={eta_value}"):
            # Reset meters
            self.loss_meters.reset()
            self.acc_meters.reset()

            recovery_model.train()

            # Batch training
            for images, labels in self.clean_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Train single batch
                log_payload = self.train_recovery_batch(
                    images, labels, recovery_model,
                    base_model, origin_param, eta_value, optimizer
                )

            # Step scheduler
            scheduler.step()

            # Evaluate on test set
            test_loss, test_acc = self.evaluate_model(recovery_model)

            # Compute L2 difference
            l2_diff = self.compute_l2_diff(recovery_model, base_model)

            # Record history
            history.append({
                'epoch': epoch + 1,
                'eta': eta_value,
                'test_accuracy': test_acc,
                'train_accuracy': self.acc_meters.avg,
                'train_loss': self.loss_meters.avg,
                'test_loss': test_loss,
                'l2_diff': l2_diff,
                'lr': optimizer.param_groups[0]['lr'],
                'model_type': model_type
            })

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}: Train Acc: {self.acc_meters.avg:.2f}%, "
                      f"Test Acc: {test_acc:.2f}%, L2 Diff: {l2_diff:.4f}")

        # Save results
        results_path = os.path.join(exp_dir, 'logs', f'recovery_{model_type}_eta_{eta_value}.json')
        with open(results_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history

    def compute_l2_diff(self, model1, model2):
        """Compute L2 difference between two models"""
        diff = 0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
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
# PART 8: Main Experiment
# ============================================================================

def main():
    """Main experiment pipeline - Eta parameter sweep"""
    print("="*60)
    print("RECOVERY EXPERIMENT - ETA PARAMETER SWEEP")
    print("="*60)

    # Setup
    exp_dir, in_colab = setup_environment()
    set_seed(42)
    device = Config.DEVICE

    print(f"Device: {device}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Eta values to test: {Config.ETA_VALUES}")

    # Load models
    print("\n" + "="*50)
    print("Loading Pre-trained Models and Noise")
    print("="*50)

    clean_model = load_model(Config.PRETRAINED_CLEAN_MODEL, device)
    poisoned_model = load_model(Config.PRETRAINED_POISONED_MODEL, device)

    if clean_model is None or poisoned_model is None:
        print("❌ Could not load required models. Please check paths.")
        return

    # Load and verify PUE noise
    try:
        pue_noise = torch.load(Config.PUE_NOISE_PATH, map_location=device)
        print(f"✅ Loaded PUE noise from {Config.PUE_NOISE_PATH}")
        print(f"   Noise contains {len(pue_noise)} class-wise noise tensors")
        for k, v in pue_noise.items():
            print(f"   Class {k}: {v.shape} tensor")
    except Exception as e:
        print(f"❌ Error loading PUE noise: {e}")
        return

    # Evaluate baseline models
    print("\n" + "="*50)
    print("Baseline Model Evaluation")
    print("="*50)

    # Prepare test data
    client_datasets, test_dataset = prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Create recovery experiment instance
    recovery_exp = EtaSweepRecovery(clean_model, poisoned_model, None, test_loader, device)

    # Evaluate baselines
    clean_loss, clean_acc = recovery_exp.evaluate_model(clean_model)
    poisoned_loss, poisoned_acc = recovery_exp.evaluate_model(poisoned_model)

    print(f"Clean Model Accuracy: {clean_acc:.2f}%")
    print(f"Poisoned Model Accuracy: {poisoned_acc:.2f}%")
    print(f"Accuracy Drop: {clean_acc - poisoned_acc:.2f}%")

    # Prepare recovery data
    print("\n" + "="*50)
    print("Preparing Recovery Data")
    print("="*50)

    recovery_size = int(Config.RECOVERY_RATE * len(client_datasets[0]))
    recovery_indices = random.sample(range(len(client_datasets[0])), recovery_size)
    recovery_dataset = Subset(client_datasets[0].dataset,
                             [client_datasets[0].indices[i] for i in recovery_indices])
    recovery_loader = DataLoader(recovery_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Using {len(recovery_dataset)} samples for recovery ({Config.RECOVERY_RATE*100}% of client 0 data)")

    # Update recovery experiment with data loader
    recovery_exp.clean_data_loader = recovery_loader

    # Run recovery experiments for all eta values
    print("\n" + "="*50)
    print("Running Recovery Experiments")
    print("="*50)

    all_recovery_results = {'poisoned': {}, 'clean': {}}

    # Recovery on poisoned model
    print("\n🔧 Recovery experiments on POISONED model:")
    for eta in tqdm(Config.ETA_VALUES, desc="Poisoned Model Recovery"):
        history = recovery_exp.run_recovery_experiment(eta, exp_dir, 'poisoned')
        all_recovery_results['poisoned'][eta] = history

    # Recovery on clean model (baseline)
    print("\n🔧 Recovery experiments on CLEAN model (baseline):")
    for eta in tqdm(Config.ETA_VALUES, desc="Clean Model Recovery"):
        history = recovery_exp.run_recovery_experiment(eta, exp_dir, 'clean')
        all_recovery_results['clean'][eta] = history

    # ============================================================================
    # Generate Comprehensive Results
    # ============================================================================

    print("\n" + "="*50)
    print("Generating Comprehensive Results")
    print("="*50)

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Final Recovery Accuracy vs Eta
    ax = axes[0, 0]
    eta_vals = Config.ETA_VALUES
    poisoned_final_accs = [all_recovery_results['poisoned'][eta][-1]['test_accuracy'] for eta in eta_vals]
    clean_final_accs = [all_recovery_results['clean'][eta][-1]['test_accuracy'] for eta in eta_vals]

    ax.plot(eta_vals, poisoned_final_accs, 'o-', label='Poisoned Model Recovery', linewidth=2, markersize=6)
    ax.plot(eta_vals, clean_final_accs, 's--', label='Clean Model Recovery', linewidth=2, markersize=6)
    ax.axhline(y=clean_acc, color='blue', linestyle=':', alpha=0.7, label='Clean Baseline')
    ax.axhline(y=poisoned_acc, color='red', linestyle=':', alpha=0.7, label='Poisoned Baseline')

    ax.set_xlabel('η (Projection Radius)')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Recovery Performance vs η')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Improvement over Poisoned Baseline
    ax = axes[0, 1]
    improvements = [acc - poisoned_acc for acc in poisoned_final_accs]
    ax.plot(eta_vals, improvements, 'ro-', linewidth=2, markersize=6)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('η (Projection Radius)')
    ax.set_ylabel('Accuracy Improvement (%)')
    ax.set_title('Recovery Improvement vs η')
    ax.grid(True, alpha=0.3)

    # Plot 3: Final L2 Distance vs Eta
    ax = axes[0, 2]
    poisoned_l2_dists = [all_recovery_results['poisoned'][eta][-1]['l2_diff'] for eta in eta_vals]
    clean_l2_dists = [all_recovery_results['clean'][eta][-1]['l2_diff'] for eta in eta_vals]

    ax.plot(eta_vals, poisoned_l2_dists, 'o-', label='From Poisoned Model', linewidth=2)
    ax.plot(eta_vals, clean_l2_dists, 's--', label='From Clean Model', linewidth=2)
    ax.plot(eta_vals, eta_vals, 'k:', alpha=0.5, label='η limit')
    ax.set_xlabel('η (Projection Radius)')
    ax.set_ylabel('Final L2 Distance')
    ax.set_title('Final L2 Distance vs η')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Recovery Trajectories for Selected Etas
    ax = axes[1, 0]
    selected_etas = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_etas)))

    for eta, color in zip(selected_etas, colors):
        if eta in all_recovery_results['poisoned']:
            epochs = [h['epoch'] for h in all_recovery_results['poisoned'][eta]]
            accs = [h['test_accuracy'] for h in all_recovery_results['poisoned'][eta]]
            ax.plot(epochs, accs, color=color, label=f'η={eta}', linewidth=2)

    ax.axhline(y=clean_acc, color='blue', linestyle=':', alpha=0.7)
    ax.axhline(y=poisoned_acc, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Recovery Trajectories (Selected η)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: L2 Distance Trajectories
    ax = axes[1, 1]
    for eta, color in zip(selected_etas, colors):
        if eta in all_recovery_results['poisoned']:
            epochs = [h['epoch'] for h in all_recovery_results['poisoned'][eta]]
            l2_dists = [h['l2_diff'] for h in all_recovery_results['poisoned'][eta]]
            ax.plot(epochs, l2_dists, color=color, label=f'η={eta}', linewidth=2)

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('L2 Distance from Poisoned Model')
    ax.set_title('L2 Distance Evolution (Selected η)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Eta Effectiveness Heatmap
    ax = axes[1, 2]

    # Create effectiveness matrix
    effectiveness_data = []
    for eta in eta_vals:
        poisoned_final = all_recovery_results['poisoned'][eta][-1]['test_accuracy']
        improvement = poisoned_final - poisoned_acc
        effectiveness_data.append(improvement)

    # Bar plot showing effectiveness
    bars = ax.bar(range(len(eta_vals)), effectiveness_data, color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(eta_vals))))
    ax.set_xlabel('η Index')
    ax.set_ylabel('Accuracy Improvement (%)')
    ax.set_title('Recovery Effectiveness by η')
    ax.set_xticks(range(len(eta_vals)))
    ax.set_xticklabels([f'{eta:.1f}' for eta in eta_vals], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, effectiveness_data)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, 'plots', 'eta_sweep_comprehensive_results.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # Generate summary
    summary = {
        'experiment_config': {
            'eta_values': Config.ETA_VALUES,
            'recovery_epochs': Config.RECOVERY_EPOCHS,
            'recovery_rate': Config.RECOVERY_RATE,
            'recovery_lr': Config.RECOVERY_LR,
            'pretrained_clean_model': Config.PRETRAINED_CLEAN_MODEL,
            'pretrained_poisoned_model': Config.PRETRAINED_POISONED_MODEL,
            'pue_noise_path': Config.PUE_NOISE_PATH
        },
        'baseline_results': {
            'clean_accuracy': clean_acc,
            'poisoned_accuracy': poisoned_acc,
            'accuracy_drop': clean_acc - poisoned_acc
        },
        'eta_sweep_results': {}
    }

    # Add detailed results for each eta
    for eta in Config.ETA_VALUES:
        poisoned_final = all_recovery_results['poisoned'][eta][-1]['test_accuracy']
        clean_final = all_recovery_results['clean'][eta][-1]['test_accuracy']
        improvement = poisoned_final - poisoned_acc

        summary['eta_sweep_results'][f'eta_{eta}'] = {
            'eta_value': eta,
            'poisoned_recovery_final': poisoned_final,
            'clean_recovery_final': clean_final,
            'improvement_over_poisoned': improvement,
            'final_l2_distance': all_recovery_results['poisoned'][eta][-1]['l2_diff']
        }

    # Find best eta
    best_eta = max(Config.ETA_VALUES,
                   key=lambda eta: all_recovery_results['poisoned'][eta][-1]['test_accuracy'])
    best_performance = all_recovery_results['poisoned'][best_eta][-1]['test_accuracy']

    summary['best_eta'] = {
        'eta_value': best_eta,
        'final_accuracy': best_performance,
        'improvement': best_performance - poisoned_acc
    }

    # Save comprehensive summary
    summary_path = os.path.join(exp_dir, 'logs', 'eta_sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print comprehensive final summary
    print("\n" + "="*60)
    print("ETA SWEEP EXPERIMENT COMPLETE!")
    print("="*60)

    print(f"\nBaseline Performance:")
    print(f"  Clean Model: {clean_acc:.2f}%")
    print(f"  Poisoned Model: {poisoned_acc:.2f}%")
    print(f"  Accuracy Drop: {clean_acc - poisoned_acc:.2f}%")

    print(f"\nEta Sweep Results (Poisoned Model Recovery):")
    print(f"{'η':>6} | {'Final Acc':>10} | {'Improvement':>12} | {'L2 Distance':>12}")
    print("-" * 50)
    for eta in Config.ETA_VALUES:
        final_acc = all_recovery_results['poisoned'][eta][-1]['test_accuracy']
        improvement = final_acc - poisoned_acc
        l2_dist = all_recovery_results['poisoned'][eta][-1]['l2_diff']
        print(f"{eta:>6.1f} | {final_acc:>9.2f}% | {improvement:>10.2f}% | {l2_dist:>11.4f}")

    print(f"\n🏆 Best Performance:")
    print(f"  Best η: {best_eta}")
    print(f"  Best Final Accuracy: {best_performance:.2f}%")
    print(f"  Best Improvement: {best_performance - poisoned_acc:.2f}%")

    print(f"\nResults saved to: {exp_dir}")
    print(f"  • Detailed logs: {exp_dir}/logs/")
    print(f"  • Comprehensive plots: {exp_dir}/plots/")
    print(f"  • Summary: {summary_path}")

    return summary

if __name__ == "__main__":
    results = main()
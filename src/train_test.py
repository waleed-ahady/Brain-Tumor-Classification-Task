#Importing all the necessary libraries
import os
import random
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
import torch
import albumentations as A
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
import time
import multiprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Configuration settings
SEED = 4030
DATA_ROOT = '/content/drive/MyDrive/classification_task'
SAVE_DIR = '/content/drive/MyDrive/Convnext_small'
NUM_CLASSES = 4
CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
PLANES = ['ax', 'co', 'sa']
BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 7
LR = 0.0001
WEIGHT_DECAY = 0.01
MIXUP_ALPHA = 0
MIXUP_PROB = 0
SCHEDULER_TYPE = 'cosine' # or 'onecycle
LABEL_SMOOTH = 0
MAX_LR = 0 #set max LR if using OneCylce


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'confusion_matrices'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'sanity_checks'), exist_ok=True)

def extract_metadata_from_filename(filename):
    clean = os.path.splitext(filename)[0].lower()
    tumor_map = {'gl': 'glioma', 'me': 'meningioma', 'pi': 'pituitary', 'no': 'no_tumor'}
    plane_map = {'ax': 'axial', 'co': 'coronal', 'sa': 'sagittal'}
    tumor_type, plane = None, None
    for code, name in tumor_map.items():
        if f'_{code}_' in clean:
            tumor_type = name
            break
    for code, name in plane_map.items():
        if f'_{code}_' in clean:
            plane = name
            break
    return tumor_type, plane

def denormalize(tensor, mean, std):
    t = tensor.clone()
    for c in range(len(mean)):
        t[c] = t[c] * std[c] + mean[c]
    return t

class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.metadata = []
        if not os.path.exists(root_dir):
            print(f"❌ Directory not found: {root_dir}")
            return
        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠️ Class directory missing: {class_dir}")
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                    path = os.path.join(class_dir, filename)
                    if not os.path.isfile(path):
                        continue
                    tumor_type, plane = extract_metadata_from_filename(filename)
                    if tumor_type and plane:
                        self.file_paths.append(path)
                        self.labels.append(CLASS_TO_IDX[tumor_type])
                        self.metadata.append((tumor_type, plane))
        print(f"Loaded {len(self.file_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)
        if self.transform:
            # Pass image as a named argument for Albumentations
            image = self.transform(image=np.array(image))['image']
        return image, self.labels[idx]
    
def compute_mean_std(dataset):
    """Compute custom mean and std for dataset normalization using GPU"""
    if len(dataset) == 0:
        print("Dataset is empty. Cannot compute mean and std.")
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    # Use same transform as validation/test (no augmentation)
    temp_transform = A.Compose([
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(256, 256, p=1.0),
        ToTensorV2(p=1.0)
    ])

    # Create loader without normalization
    temp_dataset = copy.deepcopy(dataset)
    temp_dataset.transform = temp_transform

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=min(10, multiprocessing.cpu_count()))

    # Accumulators
    mean = torch.zeros(3).to(device)
    mean_sq = torch.zeros(3).to(device)
    total_pixels = 0

    for images, _ in tqdm(loader, desc='Computing stats'):
        images = images.float().to(device) / 255.0
        batch, channels, height, width = images.shape

        # Update statistics
        n_pixels = batch * height * width
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_sq = torch.sum(images ** 2, dim=[0, 2, 3])

        mean += sum_
        mean_sq += sum_sq
        total_pixels += n_pixels

    # Compute final mean and std
    mean /= total_pixels
    std = torch.sqrt(mean_sq / total_pixels - mean ** 2)

    return mean.cpu().numpy().tolist(), std.cpu().numpy().tolist()


def build_transforms():
  train_tf= A.Compose([
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(height=256, width=256, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=[0.8, 1.2], translate_percent=[-0.05, 0.05], rotate=[-15, 15],
                  shear=[0, 0], interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                  fit_output=False, keep_ratio=False, rotate_method="ellipse", balanced_scale=True,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3),
        A.ColorJitter( brightness=[1, 2],contrast=[0.9, 1.2],saturation=[0.9, 1.2],hue=[-0.5, 0.5], p=0.3),
        A.ElasticTransform(alpha=300, sigma=8, interpolation=cv2.INTER_LINEAR, approximate=False,
                           same_dxdy=True, mask_interpolation=cv2.INTER_NEAREST, noise_distribution="gaussian",
                           keypoint_remapping_method="mask", border_mode=cv2.BORDER_CONSTANT,
                           fill=0, fill_mask=0, p=0.2),
        A.GaussNoise(std_range=[0.1, 0.2],mean_range=[0, 0],per_channel=False,noise_scale_factor=1, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(p=1.0)
    ])

  val_tf = A.Compose([
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(height=256, width=256, p=1.0),
        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(p=1.0)
  ])
  return train_tf, val_tf


def stratified_split_by_plane(dataset):
    grouped = defaultdict(lambda: defaultdict(list))
    for idx, (label, (tumor_type, plane)) in enumerate(zip(dataset.labels, dataset.metadata)):
        grouped[label][plane].append(idx)
    train_idx, val_idx = [], []
    for _, planes in grouped.items():
        for _, indices in planes.items():
            if len(indices) <= 1:
                train_idx.extend(indices)
            elif len(indices) <= 4:
                a, b = train_test_split(indices, test_size=1, random_state=SEED)
                train_idx.extend(a); val_idx.extend(b)
            else:
                a, b = train_test_split(indices, test_size=0.2, random_state=SEED)
                train_idx.extend(a); val_idx.extend(b)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def perform_sanity_checks(train_subset, val_subset, test_dataset):
    print("\nRunning sanity checks...")
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")

    base = train_subset.dataset
    def dist_fast(subset):
        counts = [0]*NUM_CLASSES
        for i in subset.indices:
            counts[base.labels[i]] += 1
        return counts

    if len(train_subset) > 0:
        idx = random.choice(list(train_subset.indices))
        pil = Image.open(base.file_paths[idx]).convert('RGB')
        tfm = base.transform
        if tfm is not None:
            aug = tfm(image=np.array(pil))['image'] # Use Albumentations transform
            aug_vis = denormalize(aug, mean, std).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            plt.imshow(aug_vis)
            plt.title(f"Class: {CLASSES[base.labels[idx]]}")
            plt.savefig(os.path.join(SAVE_DIR, 'sanity_checks', 'sample_image.png'))
            plt.close()

    print(f"Train distribution: {dist_fast(train_subset)}")
    print(f"Val distribution: {dist_fast(val_subset)}")

    plane_counts_val = defaultdict(lambda: defaultdict(int))
    plane_counts_train = defaultdict(lambda: defaultdict(int))
    for i in val_subset.indices:
        tumor_type, plane = base.metadata[i]
        plane_counts_val[tumor_type][plane] += 1
    for i in train_subset.indices:
        tumor_type, plane = base.metadata[i]
        plane_counts_train[tumor_type][plane] += 1

    print("Validation set distribution per type/plane:")
    for t in CLASSES:
        print(f"  {t}:")
        for p in PLANES:
            pname = {'ax':'axial','co':'coronal','sa':'sagittal'}[p]
            print(f"    {pname}: {plane_counts_val[t][pname]} samples")

    print("\nTraining set distribution per type/plane:")
    for t in CLASSES:
        print(f"  {t}:")
        for p in PLANES:
            pname = {'ax':'axial','co':'coronal','sa':'sagittal'}[p]
            print(f"    {pname}: {plane_counts_train[t][pname]} samples")

    tfm = base.transform
    if tfm is not None and len(train_subset) >= 2:
        idxs = random.sample(list(train_subset.indices), 2)
        imgs_orig, imgs_aug = [], []
        for i in idxs:
            pil = Image.open(base.file_paths[i]).convert('RGB')
            aug_t = tfm(image=np.array(pil))['image'] # Use Albumentations transform
            aug_vis = denormalize(aug_t, mean, std).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            imgs_orig.append(np.array(pil))
            imgs_aug.append(aug_vis)
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1); plt.imshow(imgs_orig[0]); plt.title("Original 1"); plt.axis('off')
        plt.subplot(2, 2, 2); plt.imshow(imgs_aug[0]); plt.title("Augmented 1"); plt.axis('off')
        plt.subplot(2, 2, 3); plt.imshow(imgs_orig[1]); plt.title("Original 2"); plt.axis('off')
        plt.subplot(2, 2, 4); plt.imshow(imgs_aug[1]); plt.title("Augmented 2"); plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'sanity_checks', 'aug_examples.png'))
        plt.close()

    print("Sanity checks passed!\n")


def create_model():
    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
    model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    num_ftrs = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(num_ftrs, eps=1e-6),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    return model

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


def calculate_metrics(outputs, labels):
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted')
    p_w = precision_score(y_true, y_pred, average='weighted')
    r_w = recall_score(y_true, y_pred, average='weighted')
    f1_m = f1_score(y_true, y_pred, average='macro')
    p_m = precision_score(y_true, y_pred, average='macro')
    r_m = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    per_recall = recall_score(y_true, y_pred, average=None)
    return {
        'accuracy': acc,
        'f1_weighted': f1_w,
        'precision_weighted': p_w,
        'recall_weighted': r_w,
        'f1_macro': f1_m,
        'precision_macro': p_m,
        'recall_macro': r_m,
        'confusion_matrix': cm,
        'per_class_recall': per_recall.tolist()
    }

class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_metrics = None
        self.best_epoch = None
    def __call__(self, val_loss, metrics, epoch, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, metrics, epoch, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, metrics, epoch, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, metrics, epoch, model):
        saved = {
            'accuracy': metrics['accuracy'],
            'f1': metrics.get('f1_weighted', metrics.get('f1', 0)),
            'precision': metrics.get('precision_weighted', metrics.get('precision', 0)),
            'recall': metrics.get('recall_weighted', metrics.get('recall', 0)),
            'confusion_matrix': metrics['confusion_matrix'],
            'full_metrics': metrics
        }
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_metrics = saved
        self.best_epoch = epoch
        print(f"Validation loss decreased. Saved best model at epoch {epoch+1}")

def save_final_model(model, optimizer, config, metrics, epoch, filename='final_model.pth'):
    final_metrics = {
        'accuracy': metrics['accuracy'],
        'f1': metrics.get('f1_weighted', metrics.get('f1', 0)),
        'precision': metrics.get('precision_weighted', metrics.get('precision', 0)),
        'recall': metrics.get('recall_weighted', metrics.get('recall', 0)),
        'confusion_matrix': metrics['confusion_matrix'],
        'full_metrics': metrics
    }
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': final_metrics
    }
    path = os.path.join(SAVE_DIR, filename)
    torch.save(checkpoint, path)
    print(f"Saved final model to {path}")

def save_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrices', filename))
    plt.close()

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_dir):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics', 'training_history.png'))
    plt.close()

def plot_roc_curve(outputs, labels, classes, title, filename):
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.4f})')
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-avg (AUC = {roc_auc["micro"]:.4f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-avg (AUC = {roc_auc["macro"]:.4f})',
             color='navy', linestyle=':', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {title}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'metrics', filename))
    plt.close()
    return {
        'per_class_auc': [roc_auc[i] for i in range(len(classes))],
        'macro_auc': roc_auc['macro'],
        'micro_auc': roc_auc['micro']
    }

def main():
    print("Preparing datasets...")
    train_raw = ImageClassificationDataset(os.path.join(DATA_ROOT, 'train'))
    test_dataset = ImageClassificationDataset(os.path.join(DATA_ROOT, 'test'))
    if len(train_raw) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty")
        return
    print(f"Found {len(train_raw)} training images")
    print(f"Found {len(test_dataset)} test images")


    # Compute custom mean and std
    print("Computing dataset statistics...")
    mean, std = compute_mean_std(train_raw)
    print(f"Computed mean: {mean}, std: {std}")


    train_tf, val_tf = build_transforms()
    train_raw.transform = train_tf
    test_dataset.transform = val_tf

    train_subset, val_subset = stratified_split_by_plane(train_raw)

    base = train_subset.dataset
    idxs = train_subset.indices
    train_labels_list = [base.labels[i] for i in idxs]
    counts = np.bincount(train_labels_list, minlength=NUM_CLASSES)
    total = len(train_labels_list)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    nz = counts > 0
    if nz.any():
        weights[nz] = total / (NUM_CLASSES * counts[nz])
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {dict(zip(CLASSES, class_weights.cpu().numpy()))}")
    print(f"Train size: {len(train_subset)}, Validation size: {len(val_subset)}")

    num_workers = min(10, multiprocessing.cpu_count())
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin)

    perform_sanity_checks(train_subset, val_subset, test_dataset)

    print("Creating ConvNeXt-small model...")
    model = create_model().to(device)
    unfreeze_model(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif SCHEDULER_TYPE == 'onecycle':
        assert MAX_LR > 0, "Set MAX_LR > 0 for OneCycleLR"
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=PATIENCE, path=os.path.join(SAVE_DIR, 'best_model.pt'))

    config = {
        'seed': SEED,
        'data_root': DATA_ROOT,
        'classes': CLASSES,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'mixup_alpha': MIXUP_ALPHA,
        'mixup_prob': MIXUP_PROB,
        'normalization_mean': MEAN,
        'normalization_std': STD,
        'model': 'ConvNeXt-Small',
        'optimizer': 'AdamW',
        'train_size': len(train_subset),
        'val_size': len(val_subset),
        'test_size': len(test_dataset),
        'device': str(device)
    }
    with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("Starting training...")
    best_val_metrics = None
    best_epoch = -1
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        run_loss, correct, total_seen = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - Train'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if labels.dtype == torch.long:
                loss = criterion(outputs, labels)
            else:
                loss = torch.mean(torch.sum(-labels * F.log_softmax(outputs, dim=1), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler and SCHEDULER_TYPE == 'onecycle':
                scheduler.step()
            run_loss += loss.item() * images.size(0)
            if labels.dtype == torch.long:
                preds = outputs.argmax(1)
                total_seen += labels.size(0)
                correct += (preds == labels).sum().item()
        if scheduler and SCHEDULER_TYPE == 'cosine':
            scheduler.step()
        train_loss = run_loss / len(train_loader.dataset)
        train_acc = (correct / total_seen) if total_seen > 0 else 0.0

        model.eval()
        val_loss = 0.0
        all_labels, all_outputs = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - Val'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_labels.append(labels)
                all_outputs.append(outputs)
        val_loss = val_loss / len(val_loader.dataset)
        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)
        val_metrics = calculate_metrics(all_outputs, all_labels)

        print(f'\nEpoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f' Val Loss: {val_loss:.4f} | Acc: {val_metrics["accuracy"]:.4f}')
        print(f' F1 (Weighted): {val_metrics["f1_weighted"]:.4f} | F1 (Macro): {val_metrics["f1_macro"]:.4f}')
        print(f' Macro Recall: {val_metrics["recall_macro"]:.4f} | Weighted Recall: {val_metrics["recall_weighted"]:.4f}')
        print('  Per-class Recall: ' + ', '.join([f'{c}: {r:.4f}' for c, r in zip(CLASSES, val_metrics['per_class_recall'])]))

        train_losses.append(train_loss); train_accs.append(train_acc)
        val_losses.append(val_loss); val_accs.append(val_metrics['accuracy'])

        early_stopping(val_loss, val_metrics, epoch, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            best_val_metrics = early_stopping.best_metrics
            best_epoch = early_stopping.best_epoch
            break
        if best_val_metrics is None or val_loss < early_stopping.val_loss_min:
            best_val_metrics = val_metrics
            best_epoch = epoch

    history = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': val_losses, 'val_acc': val_accs}
    with open(os.path.join(SAVE_DIR, 'metrics', 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    plot_training_history(train_losses, train_accs, val_losses, val_accs, SAVE_DIR)

    print("Training completed. Loading best model for evaluation...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pt')))
    model.eval()

    save_final_model(model, optimizer, config, best_val_metrics, best_epoch)

    val_metrics = best_val_metrics
    print("\nBest Validation Metrics:")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"F1: {val_metrics.get('f1_weighted', val_metrics.get('f1', 0)):.4f}")
    print(f"Precision: {val_metrics.get('precision_weighted', val_metrics.get('precision', 0)):.4f}")
    print(f"Recall (Sensitivity): {val_metrics.get('recall_weighted', val_metrics.get('recall', 0)):.4f}")

    save_confusion_matrix(
        val_metrics['confusion_matrix'], CLASSES,
        f'Validation Confusion Matrix (Epoch {best_epoch+1})',
        'val_confusion_matrix.png'
    )

    print("\nComputing ROC AUC for validation set...")
    val_loader_eval = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_labels_full, val_outputs_full = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader_eval, desc='Validating ROC'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_labels_full.append(labels)
            val_outputs_full.append(outputs)
    val_labels_full = torch.cat(val_labels_full)
    val_outputs_full = torch.cat(val_outputs_full)
    val_roc_metrics = plot_roc_curve(val_outputs_full, val_labels_full, CLASSES, f'Validation ROC (Epoch {best_epoch+1})', 'val_roc_curve.png')

    print("Validation ROC AUC:")
    print(f"Macro Avg: {val_roc_metrics['macro_auc']:.4f}")
    print(f"Micro Avg: {val_roc_metrics['micro_auc']:.4f}")
    print("Per-Class:")
    for cls, auc_val in zip(CLASSES, val_roc_metrics['per_class_auc']):
        print(f"  {cls}: {auc_val:.4f}")

    test_labels, test_outputs = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_labels.append(labels)
            test_outputs.append(outputs)
    test_labels = torch.cat(test_labels)
    test_outputs = torch.cat(test_outputs)
    test_metrics = calculate_metrics(test_outputs, test_labels)

    print("\nTest Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1: {test_metrics.get('f1_weighted', test_metrics.get('f1', 0)):.4f}")
    print(f"Precision: {test_metrics.get('precision_weighted', test_metrics.get('precision', 0)):.4f}")
    print(f"Recall (Sensitivity): {test_metrics.get('recall_weighted', test_metrics.get('recall', 0)):.4f}")

    save_confusion_matrix(
        test_metrics['confusion_matrix'], CLASSES,
        'Test Set Confusion Matrix', 'test_confusion_matrix.png'
    )

    print("\nComputing ROC AUC for test set...")
    test_roc_metrics = plot_roc_curve(test_outputs, test_labels, CLASSES, 'Test Set ROC', 'test_roc_curve.png')

    print("\nTest ROC AUC:")
    print(f"Macro Avg: {test_roc_metrics['macro_auc']:.4f}")
    print(f"Micro Avg: {test_roc_metrics['micro_auc']:.4f}")
    print("Per-Class:")
    for cls, auc_val in zip(CLASSES, test_roc_metrics['per_class_auc']):
        print(f"  {cls}: {auc_val:.4f}")

    metrics_report = {
        'validation': {
            'accuracy': val_metrics['accuracy'],
            'f1': val_metrics.get('f1_weighted', val_metrics.get('f1', 0)),
            'precision': val_metrics.get('precision_weighted', val_metrics.get('precision', 0)),
            'recall': val_metrics.get('recall_weighted', val_metrics.get('recall', 0)),
            'confusion_matrix': val_metrics['confusion_matrix'].tolist(),
            'roc_auc': {
                'macro': val_roc_metrics['macro_auc'],
                'micro': val_roc_metrics['micro_auc'],
                'per_class': {cls: auc for cls, auc in zip(CLASSES, val_roc_metrics['per_class_auc'])}
            }
        },
        'test': {
            'accuracy': test_metrics['accuracy'],
            'f1': test_metrics.get('f1_weighted', test_metrics.get('f1', 0)),
            'precision': test_metrics.get('precision_weighted', test_metrics.get('precision', 0)),
            'recall': test_metrics.get('recall_weighted', test_metrics.get('recall', 0)),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'roc_auc': {
                'macro': test_roc_metrics['macro_auc'],
                'micro': test_roc_metrics['micro_auc'],
                'per_class': {cls: auc for cls, auc in zip(CLASSES, test_roc_metrics['per_class_auc'])}
            }
        }
    }
    with open(os.path.join(SAVE_DIR, 'metrics', 'final_metrics.json'), 'w') as f:
        json.dump(metrics_report, f, indent=4)

    config['training_time_minutes'] = (time.time() - start_time) / 60
    config['best_epoch'] = best_epoch + 1
    with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("\nTraining completed!")
    print(f"Results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()





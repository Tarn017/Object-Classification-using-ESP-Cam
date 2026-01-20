import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import multiprocessing
from PIL import Image
import json

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

RANDOM_SEED = 42

def load_data(train_val_path, train_split, img_size, batch_size, use_amp, aug, mode):
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=aug[0]),
        transforms.RandomRotation(aug[1]),
        transforms.ColorJitter(brightness=aug[2], contrast=aug[3], saturation=aug[4]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if mode=='train':
        train_dataset = datasets.ImageFolder(root=train_val_path, transform=train_transforms)
        if train_split < 1:
            val_dataset = datasets.ImageFolder(root=train_val_path, transform=val_test_transforms)

        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        split_idx = int(train_split * dataset_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_subset = Subset(train_dataset, train_indices)
        if train_split < 1:
            val_subset = Subset(val_dataset, val_indices)

        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
    elif mode=='test':
        test_dataset = datasets.ImageFolder(root=train_val_path, transform=val_test_transforms)

    # Device-adaptive DataLoader Settings
    if use_amp:  # GPU
        num_workers = 4
        pin_memory = True
        persistent_workers = True
    else:  # CPU - aber mit Multiprocessing
        num_workers = 4  # Statt 0!
        pin_memory = False  # Kein Speedup auf CPU
        persistent_workers = True  # Worker bleiben aktiv

    if mode=='train':
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        if train_split < 1:
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )
            return train_loader, val_loader, train_dataset.class_to_idx
    elif mode=='test':
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        return test_loader, test_dataset.class_to_idx
    return train_loader, train_dataset.class_to_idx


class SimpleCNN(nn.Module):
    def __init__(self, conv_filters, full_layers, num_classes, droprate):
        super(SimpleCNN, self).__init__()

        layers = []
        # Conv Layers
        layers.extend([
            nn.Conv2d(3, conv_filters[0], kernel_size=3),
            nn.BatchNorm2d(conv_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ])
        for i in range(len(conv_filters) - 1):
            layers.extend([
                nn.Conv2d(conv_filters[i], conv_filters[i + 1], kernel_size=3),
                nn.BatchNorm2d(conv_filters[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])

        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        layers.append(nn.Flatten())

        # FC Layers
        layers.extend([
            nn.Linear(conv_filters[-1] * 4 * 4, full_layers[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ])

        for i in range(len(full_layers) - 1):
            layers.extend([
                nn.Linear(full_layers[i], full_layers[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(droprate)
            ])

        layers.append(nn.Linear(full_layers[-1], num_classes))

        # ALLES in einem Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, model_name, train_loader, num_epochs, lr, device, use_amp, dec_lr, val_loader):
    from torch.amp import GradScaler, autocast

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = GradScaler('cuda') if use_amp else None
    if dec_lr is not None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=dec_lr
        )


    best_val_acc = 0.0

    for epoch in range(num_epochs):
        if dec_lr is not None:
            current_lr = optimizer.param_groups[0]['lr']
        elif dec_lr is None:
            current_lr=lr

        print(f"\nEpoch {epoch + 1}/{num_epochs} - Learning Rate: {current_lr:.6f}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Training"):
            if use_amp:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            else:
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        if val_loader is not None:
            val_acc, val_loss = test_model(val_loader, model, criterion, device, use_amp)
            print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        if dec_lr is not None:
            scheduler.step()

        if val_loader is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_name)
                print(f"✓ Saved best model with validation accuracy: {val_acc:.2f}%")

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return model


@torch.no_grad()
def test_model(val_loader, model, criterion, device, use_amp):
    from torch.amp import autocast

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for images, labels in tqdm(val_loader, desc=f"Validation"):
        if use_amp:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        else:
            images, labels = images.to(device), labels.to(device)

        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    return val_acc, val_loss


def get_size(train_dir):
    sample_path = os.path.join(train_dir, os.listdir(train_dir)[0],
                               os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size
    print(f"Ermittelte Bildgröße der Trainingsdaten: {img_height}x{img_width}")
    return (img_height, img_width)


def get_class_names(train_dir):
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

def load_model(model_name, cnn):

    model_name=model_name+'.json'
    with open(model_name, 'r') as f:
        config = json.load(f)


    model = SimpleCNN(config['conv_filters'], config['fully_layers'], config['num_classes'], config['droprate'])

    model.load_state_dict(torch.load(config['weights']))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_size=config['img_size']
    class_names = config['class_names']
    return model, img_size, class_names

def evaluate_model(test_path, model_name, img_size=None, class_names=None, mode=0):
    import torch
    from tqdm import tqdm
    if mode==0:
        train_split = 1
        BATCH_SIZE = 32
        model, img_size, class_names = load_model(model_name)
        test_loader, class_to_idx = load_data(test_path, train_split, img_size, BATCH_SIZE, use_amp=False, aug=[0,0,0,0,0], mode='test')
    elif mode==1:
        test_loader = test_path
        model= model_name

    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in tqdm(test_loader, desc="Validation"):

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = outputs.max(1)

        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()

        preds_cpu = predicted.detach().to("cpu")
        labels_cpu = labels.detach().to("cpu")

        for c in range(num_classes):
            tp[c] += ((preds_cpu == c) & (labels_cpu == c)).sum()
            fp[c] += ((preds_cpu == c) & (labels_cpu != c)).sum()
            fn[c] += ((preds_cpu != c) & (labels_cpu == c)).sum()

    val_acc = 100.0 * val_correct / val_total

    tp = tp.to(torch.float32)
    fp = fp.to(torch.float32)
    fn = fn.to(torch.float32)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # dict: {class_name: {"precision":..., "recall":..., "f1":...}}
    per_class_metrics = {
        class_names[i]: {
           # "precision": float(precision[i].item()),
           # "recall": float(recall[i].item()),
            "f1": float(f1[i].item()),
        }
        for i in range(num_classes)
    }
    width = max(len(name) for name in class_names)
    lines = ["F1 Score per class:"]
    for name in class_names:
        f1 = per_class_metrics[name]["f1"]
        lines.append(f"  {name:<{width}}  {f1:.{3}f}")
    f1_score= "\n".join(lines)
    print(f"Accuracy: {val_acc}")
    print(f1_score)
    return val_acc, val_loss, f1_score

def CNN(train_path, epochs, lr, conv_filters, fully_layers, resize, model_name, train_split, droprate=0, augmentation=[0,0,0,0,0], dec_lr=None):
    vorname=model_name
    model_name=model_name+'.pth'
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device}")
    if use_amp:
        print("Mixed Precision Training: ENABLED")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Mixed Precision Training: DISABLED (CPU mode)")

    # Hyperparameter
    BATCH_SIZE = 32
    #lr = 0.005
    #epochs = 4
    #train_split = 0.9

    #train_path = "./train_val/"
    img_size = get_size(train_path)
    print(img_size)
    if resize != None:
        img_size = resize
        print(f"resize images to {img_size}")

    if train_split<1:
        train_loader, val_loader, class_to_idx = load_data(train_path, train_split, img_size, BATCH_SIZE, use_amp, aug=augmentation, mode='train')
    else:
        train_loader, class_to_idx = load_data(train_path, train_split, img_size, BATCH_SIZE, use_amp, aug=augmentation, mode='train')


    class_names = get_class_names(train_path)
    num_classes=len(class_names)
    print("Class Labels:")
    print(class_names)

    model = SimpleCNN(conv_filters, fully_layers, num_classes=num_classes,droprate=droprate)
    model = model.to(device)
    print(model)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print("\nStarting training...")
    if train_split<1:
        model = train_model(model, model_name, train_loader, epochs, lr, device, use_amp, dec_lr, val_loader)
        model.load_state_dict(torch.load(model_name))
        test_acc, test_loss, test_f1 = evaluate_model(val_loader, model, img_size, class_names, mode=1)
    else:
        model = train_model(model, model_name, train_loader, epochs, lr, device, use_amp, dec_lr, val_loader=None)

    config = {
        "conv_filters": conv_filters,
        "fully_layers": fully_layers,
        "num_classes": num_classes,
        "weights": model_name,
        "img_size": img_size,
        "class_names": class_names,
        "droprate": droprate,
    }

    with open(vorname+'.json', 'w') as f:
        json.dump(config, f, indent=2)


class SimpleFFN(nn.Module):
    def __init__(self, fully_layers, num_classes, droprate, img_size, in_channels=3, use_bn=True):
        super().__init__()

        H, W = img_size
        in_features = in_channels * H * W

        layers = [nn.Flatten()]

        prev = in_features
        for h in fully_layers:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(droprate))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def FFN(train_path, epochs, lr, fully_layers, resize, model_name, train_split, droprate=0, augmentation=[0,0,0,0,0], dec_lr=None):
    vorname=model_name
    model_name=model_name+'.pth'
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device}")
    if use_amp:
        print("Mixed Precision Training: ENABLED")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Mixed Precision Training: DISABLED (CPU mode)")

    # Hyperparameter
    BATCH_SIZE = 32
    #lr = 0.005
    #epochs = 4
    #train_split = 0.9

    #train_path = "./train_val/"
    img_size = get_size(train_path)
    print(img_size)
    if resize != None:
        img_size = resize
        print(f"resize images to {img_size}")

    if train_split<1:
        train_loader, val_loader, class_to_idx = load_data(train_path, train_split, img_size, BATCH_SIZE, use_amp, aug=augmentation, mode='train')
    else:
        train_loader, class_to_idx = load_data(train_path, train_split, img_size, BATCH_SIZE, use_amp, aug=augmentation, mode='train')


    class_names = get_class_names(train_path)
    num_classes=len(class_names)
    print("Class Labels:")
    print(class_names)

    model = SimpleFFN(fully_layers, num_classes=num_classes,droprate=droprate, img_size=img_size)
    model = model.to(device)
    print(model)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print("\nStarting training...")
    if train_split<1:
        model = train_model(model, model_name, train_loader, epochs, lr, device, use_amp, dec_lr, val_loader)
        model.load_state_dict(torch.load(model_name))
        test_acc, test_loss, test_f1 = evaluate_model(val_loader, model, img_size, class_names, mode=1)
    else:
        model = train_model(model, model_name, train_loader, epochs, lr, device, use_amp, dec_lr, val_loader=None)

    config = {
        "conv_filters": None,
        "fully_layers": fully_layers,
        "num_classes": num_classes,
        "weights": model_name,
        "img_size": img_size,
        "class_names": class_names,
        "droprate": droprate,
    }

    with open(vorname+'.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    """
    CNN(
        train_path="./klass_daten/",
        epochs=10,
        lr=0.005,
        conv_filters=[16, 32, 64, 128],
        fully_layers=[256],
        resize=(128, 128),
        model_name='peter',
        train_split=0.9,
        droprate=0.5,
        augmentation=[0.1,0.1,0.1,0.1,0.1],
        dec_lr=10e-5
    )

    test_acc, test_loss, test_f1 = evaluate_model(test_path='./klass_daten/', model_name='peter')
    """
    FFN(
        train_path="./klass_daten/",
        epochs=15,
        lr=0.001,
        fully_layers=[1000,1000,1000],
        resize=(65, 65),
        model_name='peter',
        train_split=0.9,
        droprate=0.5,
        augmentation=[0.1, 0.1, 0.1, 0.1, 0.1],
        dec_lr=10e-5
    )

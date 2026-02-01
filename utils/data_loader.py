from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import configs.config as config


def get_transforms():
    """Get training and validation transforms"""

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    return train_transform, val_transform


def get_dataloaders(data_path=None):
    """Create train, validation, and test dataloaders"""

    if data_path is None:
        data_path = config.DATA_PATH

    train_transform, val_transform = get_transforms()

    # Load datasets
    train_data = datasets.ImageFolder(data_path + '/train', transform=train_transform)
    val_data = datasets.ImageFolder(data_path + '/valid', transform=val_transform)
    test_data = datasets.ImageFolder(data_path + '/test', transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    print(f"Train: {len(train_data)} images")
    print(f"Val: {len(val_data)} images")
    print(f"Test: {len(test_data)} images")

    return train_loader, val_loader, test_loader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406] # per normalizzare
IMAGENET_STD = [0.229, 0.224, 0.225] # per normalizzare

def get_dataloaders(datasets, batch_size):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    #per validazione e testing uso la stessa trasformazione siccome non ho bisogno di data augmentation
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = datasets.ImageFolder("datasets/train", train_transform)
    val_set = datasets.ImageFolder("datasets/val", test_transform)
    test_set = datasets.ImageFolder("datasets/test", test_transform)

    #da verificare se posso utilizzare 4 workers
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
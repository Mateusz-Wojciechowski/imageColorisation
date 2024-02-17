from torchvision import transforms
from GrayColorDataset import GrayColorDataset
import torch.utils.data as data
from torch.utils.data import DataLoader
from constants import BATCH_SIZE

transform = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.ToTensor()
])

dataset = GrayColorDataset(color_dir='Images/color', gray_dir='Images/gray', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

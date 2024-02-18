from torchvision import transforms
from GrayColorDataset import GrayColorDataset
import torch.utils.data as data
from torch.utils.data import DataLoader
from constants import BATCH_SIZE
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.ToTensor()
])

torch.manual_seed(0)
# dataset = GrayColorDataset(color_dir=r'Images/color', gray_dir=r'Images/gray', transform=transform)
dataset = GrayColorDataset(color_dir=r'/content/my_data/landscape Images/color', gray_dir=r'/content/my_data/landscape Images/gray', transform=transform)

train_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# for org, target in test_loader:
#     image = org[0]
#     target_im = target[0]
#     break
#
# plt.imshow(image.permute(1, 2, 0), cmap='gray')
# plt.show()
#
# plt.imshow(target_im.permute(1, 2, 0))
# plt.show()


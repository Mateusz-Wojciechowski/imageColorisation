import os
from PIL import Image
from torch.utils.data import Dataset

class GrayColorDataset(Dataset):
    def __init__(self, color_dir, gray_dir, transform=None):
        self.color_dir = color_dir
        self.gray_dir = gray_dir
        self.transform = transform
        self.filenames = os.listdir(gray_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        gray_path = os.path.join(self.gray_dir, self.filenames[idx])
        color_path = os.path.join(self.color_dir, self.filenames[idx])

        grayscale_image = Image.open(gray_path).convert('L')
        color_image = Image.open(color_path).convert('RGB')

        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            color_image = self.transform(color_image)

        return grayscale_image, color_image


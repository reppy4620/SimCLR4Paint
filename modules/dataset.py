import glob

from PIL import Image
from torch.utils.data import Dataset


# Normal dataset for image
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = glob.glob(f'{root}/*')
        self.length = len(self.data)
        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.data[idx]).convert('RGB')
        except:
            img = Image.open(self.data[idx-3]).convert('RGB')
        img = self.transform(img)
        return img
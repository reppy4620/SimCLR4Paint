import glob

from PIL import Image
from torch.utils.data import Dataset


# Normal dataset for image
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = glob.glob(f'{root}/*.jpg').extend(glob.glob(f'{root}/*.png'))
        self.length = len(self.data)
        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.data[idx]).convert('RGB')
        except:
            # lazy exception handling
            # if occur errors because of image loading, fix this line.
            img = Image.open(self.data[idx-3]).convert('RGB')
        img = self.transform(img)
        return img

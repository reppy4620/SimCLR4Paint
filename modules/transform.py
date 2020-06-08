import torchvision.transforms as T


class TransformsSimCLR:

    def __init__(self, size):
        # ColorJitter always transforms images by using range of arguments.
        # so with the use of RandomApply module, make ColorJitter transform 0.8 probability
        color_jitter = T.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2
        )
        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(p=0.5),  # default value
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                # normalizing stats are employed from https://github.com/RF5/danbooru-pretrained
                # because of transfer learning.
                T.Normalize(mean=[0.7137, 0.6628, 0.6519],
                            std=[0.2970, 0.3017, 0.2979])
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

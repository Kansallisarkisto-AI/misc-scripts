import random
import torchvision.transforms.v2 as T
import io
from PIL import Image

class RandomJPEGCompression:
    """Compress image to JPEG with random quality."""
    
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)

        quality = random.randint(*self.quality_range)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert(img.mode)

        return compressed


class OwnAug:
    """Randomly chosen image augmentations."""

    def __init__(self):
        # Augmentations for distorting image
        self.perspective_transformer = T.RandomPerspective(distortion_scale=0.2, p=1.0, fill = 255)
        self.affine_transfomer = T.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.10), scale=(0.7, 1.05), fill=255)
        self.elastic_transformer = T.ElasticTransform(alpha=7.0, fill=255)
        self.img_transforms = T.RandomChoice([self.perspective_transformer, self.affine_transfomer, self.elastic_transformer])

        # Augmentations for changing colors
        self.jitter = T.ColorJitter(brightness=.5, hue=.3, contrast = 0.2, saturation = 0.2)
        self.equalizer = T.RandomEqualize(p=1.0)
        self.color_transforms = T.RandomChoice([self.jitter, self.equalizer], p=[0.8,0.2])

        # Augmentation for blurring
        self.blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))

        # Augmentation for compressing as JPEG
        self.jpeg_compressor = RandomJPEGCompression(
            quality_range=(60, 98)
        )

    def __call__(self, img):
        if random.random() < 0.1:
            img = self.img_transforms(img)
        if random.random() < 0.2:
            img = self.color_transforms(img)
        if random.random() < 0.1:
            img = self.blurrer(img)
        if random.random() < 0.1:
            img = self.jpeg_compressor(img)
        
        return img


class OwnHeavyAug:
    """Randomly chosen HEAVY image augmentations."""

    def __init__(self):
        # Augmentations for distorting image
        self.perspective_transformer = T.RandomPerspective(distortion_scale=0.2, p=1.0, fill = 255)
        self.affine_transfomer = T.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.10), scale=(0.7, 0.80), fill=255)
        self.elastic_transformer = T.ElasticTransform(alpha=7.0, fill=255)
        self.img_transforms = T.RandomChoice([self.perspective_transformer, self.affine_transfomer, self.elastic_transformer])

        # Augmentations for changing colors
        self.jitter = T.ColorJitter(brightness=.5, hue=.3, contrast = 0.2, saturation = 0.2)
        self.equalizer = T.RandomEqualize(p=1.0)
        self.color_transforms = T.RandomChoice([self.jitter, self.equalizer], p=[0.8,0.2])

        # Augmentation for blurring
        self.blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))

        # Augmentation for compressing as JPEG
        self.jpeg_compressor = RandomJPEGCompression(
            quality_range=(60, 95)
        )

    def __call__(self, img):
        if random.random() < 0.3:
            img = self.img_transforms(img)
        if random.random() < 0.3:
            img = self.color_transforms(img)
        if random.random() < 0.3:
            img = self.blurrer(img)
        if random.random() < 0.3:
            img = self.jpeg_compressor(img)
        
        return img

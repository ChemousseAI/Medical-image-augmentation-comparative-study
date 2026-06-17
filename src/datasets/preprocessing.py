
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

def normalize_images(images):
    return [transform(img) for img in images]

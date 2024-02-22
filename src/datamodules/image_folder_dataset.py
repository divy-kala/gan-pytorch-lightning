import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob 
import PIL
from PIL import Image
from glob import glob

class ImageFolderClassification(Dataset):
    '''
    Creates a dataset from root_dir, which is supposed to have two folders 'employee' and 'customer'
    __getitem__ returns, if no errors, (preprocessed image, "YES", image name, 1 for employee and 0 for customer)
    preprocess function should return a numpy array, tensor, number, etc., otherwise default collate_fn will not work in DataLoader
    In main func(), code for visualizing the images after preprocessing is also given
    
    '''
    def __init__(self, data_root: str, preprocess= lambda x:x):
        print(f'building dataset from {data_root} ...')
        self.data_root = data_root
        self.all_paths = sorted(glob(os.path.join(self.data_root, '*/*')))
        self.preprocess = preprocess
        self.dummy = torch.zeros(3, 224, 224)

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index: int):
        fname = self.all_paths[index]
        image_path = f"{fname}"
        assert os.path.exists(image_path)
        is_error = False

        label = 0 if os.path.normpath(image_path).split(os.path.sep)[-2] == 'customer' else 1

        image = self.dummy
        try:
            image = self.preprocess(Image.open(image_path))
        except PIL.UnidentifiedImageError as e:
            is_error = True
            print(e)
        except OSError as e:
            is_error = True
            print(e)
        except BaseException as e:
            is_error = True
            print(e)
        if is_error:
            return image, "ERROR", os.path.basename(image_path), label
        return image, 'YES', os.path.basename(image_path), label
    

if __name__ == '__main__':
    ds = ImageFolderClassification('data/output')
    print(ds[0])
    print(ds[1])
    

    # Test with ResNet preprocessing
    from torchvision import transforms
    transform = transforms.Compose([
            transforms.Resize(size=(640,640)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ds = ImageFolderClassification('data/output', preprocess=transform)
    img = ds[0][0]
    
    # Reverse ResNet preprocessing and save image for dataset and augmentation visualization
    reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Lambda(lambda x: x.clamp(0, 1)),  # Clamp to ensure valid pixel values
            transforms.ToPILImage()
        ])
    reverse_transform(img).save('hehe.jpg')
    


    # Creates a dataset with padded images while maintaining aspect ratio, and reverses the transformation for output
    import cv2
    import numpy as np
    
    class ResizeWithPadding:
        def __call__(self, image, target_size=(224,224)):
    
            image = np.array(image)
            h, w = image.shape[:2]
            target_w, target_h = target_size
            aspect_ratio = w / h
            target_aspect_ratio = target_w / target_h
    
            if aspect_ratio > target_aspect_ratio:
                new_w = target_w
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = target_h
                new_w = int(new_h * aspect_ratio)
    
            resized_img = cv2.resize(image, (new_w, new_h))
    
            pad_w = (target_w - new_w) 
            pad_h = (target_h - new_h)
    
            padded_img = np.pad(resized_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            padded_img = Image.fromarray(np.uint8(padded_img)).convert('RGB')
    
            return padded_img
        
    # Test with ResNet preprocessing + padded image with maintained aspect ratio
    from torchvision import transforms
    transform = transforms.Compose([
            ResizeWithPadding(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
        ])
    ds = ImageFolderClassification('data/output', preprocess=transform)
    img = ds[0][0]
    
    # Reverse ResNet preprocessing and save image for dataset and augmentation visualization
    reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Lambda(lambda x: x.clamp(0, 1)),  # Clamp to ensure valid pixel values
            transforms.ToPILImage()
        ])
    reverse_transform(img).save('hehe_aspect.jpg')


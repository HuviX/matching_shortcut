from torchvision import transforms
from PIL import Image
from os import listdir


#Класс для подготовки данных
class NeighborLoader:
    
    def __init__(self):
        pass
    
    def class2name(self, cl):
        return self._class2name[cl]
    
    def name2class(self, name):
        return self._name2class[name]
    
    def load(self, path):
        """
        Parameters:
        ------------
        path: str
            Path to folder with images
        Returns:
        ------------
           :(list, list)
           (List of numpy vectors, list of labels)
        """

        names = listdir(path)
        name2class = {}
        class2name = {}
        imgs = []
        labels = []
        for i, name in enumerate(names):
            name2class[name] = i
            class2name[i] = name
            img = NeighborLoader.transform(path+'//'+name) 
            imgs.append(img)
            labels.append(i)
        self._name2class = name2class
        self._class2name = class2name
        return imgs, labels
    
    
    @staticmethod
    def transform(path: str):

        """
        Parameters:
        ------------
        path: str
            Path to image to transform
        Returns:
        ------------
           :pytorch.tensor
            Transformed image
        """

        data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]),
    ])
        img = Image.open(path).convert('RGB')   
        return data_transforms(img)

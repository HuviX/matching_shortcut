from torchvision import transforms, datasets, models
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


#import faiss
from scipy.spatial.distance import cosine as cos_dist
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from loader import NeighborLoader


class BaseModel(nn.Module):
    """
    Base model to use in Matcher class.
    Use for image vectorization.

    """
    def __init__(self, model='resnet'):
        super().__init__()
        if model == 'resnet':
            self.model_type = model
            self.model = models.resnet34(pretrained=True)
            # remove last layer to avoid forward hook using
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        else:
            self.model_type = 'effnet'
            self.model = EfficientNet.from_pretrained('efficientnet-b3')
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.eval()


    def forward(self, img):
        """
        Parameters:
        ------------
        img: torch.Tensor


        Returns:
        : numpy.array()
            Image embedding from a neural net
        """

        with torch.no_grad():
            if self.model_type != 'resnet':
                feats = self.model.extract_features(img.unsqueeze(0))
                return self.avg_pool(feats).squeeze().numpy()
            return self.model(img.unsqueeze(0)).squeeze().numpy()


class Matcher:
    def __init__(
        self,
        params: dict,
        imgs: list,
        labels: list,
        vectorizer,
        loader,
):

        """
        Parameters:
        ------------
        params: dict 
            dictionary of KNeighborsClassifier params
        imgs: list 
            list of pytorch tensors
        labels: list 
            list of image labels
        vectorizer: BaseModel instance
            Transform input image (from list of imgs) to numpy vector.
        """

        self.vectorizer = vectorizer
        self.loader = loader
        self.model = KNeighborsClassifier(**params)
        self.__fill_Xy(imgs, labels)
        self.__fit()
    
    def __fill_Xy(self, imgs: list, labels: list):
        """
        Fills X and y data.
        X corresponds to vectorized images
        y corresponds to labels of vectorized images


        Parameters:
        ------------
        imgs: list 
            list of pytorch tensors
        labels: list 
            list of image labels

        Returns:
        ------------
            None
        """
        
        X, y = [], []
        with torch.no_grad():
            for i, image in enumerate(imgs):
                out = self.vectorizer(image)
                X.append(out)
                y.append(labels[i])
        self.X = np.array(X)
        self.y = np.array(y)

    def __fit(self):
        self.model.fit(self.X, self.y)
        
    
    def predict(self, path: str, tr=0.85, visualize=True):
        """
        Makes a prediction for passed image (path of the image)


        Parameters:
        ------------
        path: str 
            Path to image
        tr: float 
            threshold value.
        visuzalize: bool
            if True pairs are visualized
        
        
        Returns:
        ------------
            :list
            list of similar images
        """
        answer_paths = 'data/B/' if 'A' in path else 'data/A/'
        
        x = NeighborLoader.transform(path)
        x = self.vectorizer(x).reshape(1, -1)
        probs = self.model.predict_proba(x)
        indices = np.nonzero(probs)[1]
        
        imgs = []
        ret = []
        for i in indices:
            cd = (1 - cos_dist(self.X[i], x))
            if cd > tr:
                img_name = self.loader.class2name(i)
                ret.append(img_name)
                if visualize:
                    img = Image.open(answer_paths+img_name).convert('RGB')
                    imgs.append(img)
        
        if len(imgs) > 0:
            print("match")
            for i, im in enumerate(imgs):
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(im)
                ax[1].imshow(plt.imread(path))
                plt.show()
            print("*"*10)
        return ret
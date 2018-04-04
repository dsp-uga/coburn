"""
The preprocess module defines Transform classes which can be used to preprocess data as it is retireved from a Dataset.
Each transform class should extend coburn.data.Transform.
Transforms can be composed together using torchvision.transforms.Compose
"""

from .Transform import Transform
import os;
from skimage.transform import resize
from skimage.io import imshow,imread,imsave
import numpy as np
from PIL import Image
import thunder as td


class Mean(Transform):
    """
    Computes the mean of a series of images along the time axis
    """

    def __call__(self, images):
        return images.mean()


class Variance(Transform):
    """
    Computes the variance of a series of images along the time axis
    """

    def __call__(self, images):
        return images.var()


class Gaussian(Transform):
    """
    Computes the gaussian smoothing of images along the time axis
    """

    def __init__(self, sigma, size):
        self.size = size;
        self.sigma = sigma

    def __call__(self, images):
        return images.gaussian_filter(sigma=self.sigma, order=self.size);


class Median(Transform):
    """
    Computes the median filter of smoothing the images along the time axis
    """

    def __init__(self, size):
        self.size = size;

    def __call__(self, images):
        return images.median_filter(size=self.size);


class Deviation(Transform):
    """
    Computes the standard deviation of images along the time axis
    """

    def __call__(self, images):
        return images.std();


class Subtract(Transform):
    """
    This will subtract given value from all the images

    """

    def __init__(self, size):
        self.size = size;

    def __call__(self, images):
        return images.subtract(val=self.size);


class UniformFilter(Transform):
    """
    Applies uniform filter to all the images

    """

    def __init__(self, size):
        self.size = size;

    def __call__(self, images):
        return images.uniform_filter(size=self.size);

class Resize(Transform) :

    """
    This will resize the images and masks to the desired value
    Method :- init
    @:param dataset :- takes in a dataset object which can be used to get image details
    @:param shape1,shape2 :- New dimesions of image to resize
    @:param resize :- a boolean value which lets you know if resize is required
    @:param baseDir :- path to access data

    Method :- store_resized_images
             This method basically loads all the images and masks for each hash resizes them
             and stores the results in new folder
    @:param train :- if true then resizes masks as well meaning that we are dealing with training data

    """

    def __init__(self, dataset,shape1,shape2,resize,baseDir='data/'):

        self.shape1=shape1
        self.shape2=shape2
        self.dataset=dataset
        self.resize=resize
        self.baseDir=baseDir


    def __call__(self, images):


        img_arr = images.toarray()
        resArr=[]
        for x in range(len(img_arr)):
            rs=Image.fromarray(img_arr[x])
            rsim=rs.resize((self.shape1,self.shape2))
            reimArr=np.array(rsim)
            res = np.pad(reimArr, ((1, 1), (1, 1)), 'constant')
            resArr.append(res)
        images=td.images.fromarray(resArr)


        return images

    def store_resized_images(self,train):
        for i in range(len(self.dataset)):
            hash=self.dataset.get_hash(i)
            imagepath=os.path.join(self.baseDir,hash,'images')


            resizedpath = os.path.join(self.baseDir,hash, 'resized')
            msk_arr = []
            msk_png_resize=None
            if train :
                maskpath = os.path.join(self.baseDir, hash, 'mask.png')
                msk_png = imread(maskpath)
                msk_png_resize = resize(msk_png, (self.shape1, self.shape2))
                msk_png_resize = np.pad(msk_png_resize, ((1, 1), (1, 1)), 'constant')
                resized_mask_path = os.path.join(resizedpath, 'mask.png')
                imsave(resized_mask_path, msk_png_resize)

            if not os.path.exists(resizedpath):
                os.makedirs(resizedpath)
            images=os.listdir(imagepath)

            img_arr=[]

            for image in images:
                img=os.path.join(imagepath,image)
                org_img=imread(img)
                resized_img=resize(org_img,(self.shape1,self.shape2))
                resized_img = np.pad(resized_img, ((1, 1), (1, 1)), 'constant')
                resized_img_path=os.path.join(resizedpath,'images')
                if not os.path.exists(resized_img_path):
                    os.makedirs(resized_img_path)
                resizedimagepath=os.path.join(resized_img_path,image)
                img_arr.append(resized_img)
                if train:
                    msk_arr.append(msk_png_resize)



                imsave(resizedimagepath,resized_img)
            np.save(os.path.join(resizedpath,hash+"-train-img.npy"),img_arr)
            if train :
                np.save(os.path.join(resizedpath, hash + "-trainmask.npy"), msk_arr)


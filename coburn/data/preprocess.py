"""
The preprocess module defines Transform classes which can be used to preprocess data as it is retireved from a Dataset.
Each transform class should extend coburn.data.Transform.
Transforms can be composed together using torchvision.transforms.Compose
"""

from .Transform import Transform
import os
import cv2
from skimage.transform import resize
from skimage.io import imshow,imread,imsave
import numpy as np
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
        self.size = size
        self.sigma = sigma

    def __call__(self, images):
        return images.gaussian_filter(sigma=self.sigma, order=self.size)


class Deviation(Transform):
    """
    Computes the standard deviation of images along the time axis
    """

    def __call__(self, images):
        return images.std()


class Subtract(Transform):
    """
    This will subtract given value from all the images
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        return images.subtract(val=self.size)


class UniformFilter(Transform):
    """
    Applies uniform filter to all the images
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        return images.uniform_filter(size=self.size)


class MedianFilter(Transform):
    """
    Applies a median filter with the specified kernel size to every image in the series
    """
    def __init__(self, size):
        self.size = size or 2

    def __call__(self, images):
        return images.median_filter(size=self.size)


class OpticalFlow(Transform):
    """
    Computes optical flow between each sequential pair of images.
    return shape has an extra dimension of size 2, and the first dimension is
    of size one less (e.g. 100 becomes 99, i.e. shape[0]-=1).
    """

    def __call__(self, images):
        images = np.array(images)
        flows = np.empty((0, images.shape[1], images.shape[2], 2))
        i = 0
        while(i < images.shape[0]-1):
            flow = cv2.calcOpticalFlowFarneback(
                images[i], images[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows = np.append(flows, [flow], axis=0)
            i += 1
        return td.images.images.Images(flows)


class Magnitude(Transform):
    """
    Computes the l2 norms of each image pixelwise across multiple channels.
    Squashes and eliminates the channel dimension,
    useful for example in thresholding based on optical flow
    """

    def __call__(self, images):
        images = images.toarray()
        l2_norms = np.linalg.norm(images, ord=2, axis=3)
        return td.images.images.Images(l2_norms)


class Resize(Transform):

    """
    This will resize the images and masks to the desired value
    Method :- init
    @:param dataset :- takes in a dataset object which can be used to get image details
    @:param width,height :- New dimesions of image to resize
    @:param resize :- a boolean value which lets you know if resize is required
    @:param baseDir :- path to access data

    Method :- store_resized_images
             This method basically loads all the images and masks for each hash resizes them
             and stores the results in new folder


    """

    def __init__(self, dataset,width,height,baseDir='data/'):

        self.width=width
        self.height=height
        self.dataset=dataset
        self.resize=resize
        self.baseDir=baseDir

    def __call__(self, images):


        img_arr = images.toarray()
        resArr=[]
        for x in range(len(img_arr)):
            imgarr=np.array(img_arr[x])
            img=np.resize(imgarr,(self.width,self.height))
            resArr.append(img)

        images=td.images.fromarray(resArr)
        print(images.shape)


        return images

    def store_resized_images(self):
        for i in range(len(self.dataset)):
            hash=self.dataset.get_hash(i)
            imagepath=os.path.join(self.baseDir,hash,'images')


            resizedpath = os.path.join(self.baseDir,hash, 'resized')
            msk_arr = []


            maskpath = os.path.join(self.baseDir, hash, 'mask.png')
            msk_png = imread(maskpath)
            msk_png_resize = resize(msk_png, (self.width, self.height))
            resized_mask_path = os.path.join(resizedpath, 'mask.png')
            imsave(resized_mask_path, msk_png_resize)

            if not os.path.exists(resizedpath):
                os.makedirs(resizedpath)
            images=os.listdir(imagepath)

            img_arr=[]

            for image in images:
                img=os.path.join(imagepath,image)
                org_img=imread(img)
                resized_img=resize(org_img,(self.width,self.height))
                resized_img_path=os.path.join(resizedpath,'images')
                if not os.path.exists(resized_img_path):
                    os.makedirs(resized_img_path)
                resizedimagepath=os.path.join(resized_img_path,image)
                img_arr.append(resized_img)
                msk_arr.append(msk_png_resize)



                imsave(resizedimagepath,resized_img)
            outpath='content'
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            np.save(os.path.join(outpath,hash+"-train-img.npy"),img_arr)
            np.save(os.path.join(outpath, hash + "-train-mask.npy"), msk_arr)


class Padding(Transform):
    '''
    This method will add uniform padding for the image and return the zero padded resultant image
    '''

    def __init__(self,top,bottom,left,right):
        self.top=top
        self.bottom=bottom
        self.left=left
        self.right=right

    def __call__(self, images):
        img_arr = images.toarray()
        resArr=[]
        for x in range(len(img_arr)):
            curr_img=np.array(img_arr[x])
            padded_img = np.pad(curr_img, ((self.top, self.bottom), (self.left, self.right)), 'constant')
            resArr.append(padded_img)
        images=td.images.fromarray(resArr)
        return images

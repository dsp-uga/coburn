"""
The preprocess module defines Transform classes which can be used to preprocess data as it is retireved from a Dataset.
Each transform class should extend coburn.data.Transform.
Transforms can be composed together using torchvision.transforms.Compose
"""

from skimage.transform import resize as sk_resize
import thunder as td
import numpy as np
from .Transform import Transform
import os;
from skimage.transform import resize
from skimage.io import imshow,imread,imsave
import numpy as np
import thunder as td


class UniformResize(Transform):
    """
    Resizes the series of images to be of uniform size
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, images):
        images = images.toarray()
        resized = np.zeros((len(images), self.height, self.width))
        for idx in range(0, len(images)):
            img = images[idx]
            resized_img = sk_resize(img, (self.height, self.width))
            resized[idx] = resized_img

        return td.images.fromarray(resized)


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

class Resize(Transform) :

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
        
class ToArray(Transform):
    """
    Converts a thunder.Images object to a numpy ndarray with dimensions [H x W x T]
    where H is the height, W is the width, and T is the time or number of channels
    """
    def __call__(self, images):
        # handle the special case when the array is 2D:
        images = images.toarray()
        if len(images.shape) == 2:
            images = images[:, :, np.newaxis]
            return images

        return images.swapaxes(0, 2).swapaxes(0, 1)  # move the non-spatial axis to the correct position


class MaskToSegMap(Transform):
    """
    Converts an m x n PNG mask to a segmentation map.  Mask should be a numpy array with shape (m, n)
    A segmentation map will be m x n x 3.
    segmap[row, col, i] will be 1 if the mask has class i at location (row, col)
    """
    def __call__(self, mask):
        shape = mask.shape
        segmap = np.empty((shape[0], shape[1], 3))

        segmap[mask == 2] = [0, 0, 1]
        segmap[mask == 1] = [0, 1, 0]
        segmap[mask == 0] = [1, 0, 0]
        return segmap


class ResizeMask(Transform):
    """
    Resizes a PNG mask to be the specified size
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, mask):
        return sk_resize(mask, (self.height, self.width))

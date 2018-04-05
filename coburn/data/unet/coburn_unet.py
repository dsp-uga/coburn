from skimage.transform import resize
import numpy as np
from skimage.io import imshow
# importing keras function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

# import our model
from coburn.experiments.unet   import unet

class coburn_unet_model():
    '''
    This class is basically used to load the unet model.
    Takes the numpy files of images and masks which were preprocessed
    method :- fit
    This method is used for training the model
    method :- predict
    This method is used for testing the unet model
    '''

    def __init__(self,imageLoc,maskLoc):
        self.imageLoc=imageLoc
        self.maskLoc=maskLoc;


    def fit(self):

        images = np.load(self.imageLoc)
        masks = np.load(self.maskLoc)
        train = images[..., np.newaxis]
        masks = masks[..., np.newaxis]
        train = train.astype('float32')
        mask_images = masks.astype('float32')
        model = unet()
        model.fit(train, mask_images, batch_size=8, epochs=10, verbose=1, shuffle=True)
        model.save("model.h5")


    # predict method to use saved model to predict mask of test images
    def predict(self):
        images = np.load(self.path_image)
        test = images[..., np.newaxis]
        test = test.astype('float32')
        model = unet()
        model.load_weights('model.h5')
        predicted = model.predict(test, verbose=1)
        np.save('mask.npy', predicted)


if __name__ == '__main__':

     model =coburn_unet_model("train.npy","mask.npy")
     model.fit()






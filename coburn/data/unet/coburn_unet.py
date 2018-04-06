import numpy as np


from coburn.experiments.unet   import unet
import argparse


'''
This model was taken into consideration from Proj 3 Team Canady which was developed by our team member Vibodh and his group
UNET paper -https://arxiv.org/pdf/1707.06314.pdf
'''
class coburn_unet_model():
    '''
    This class is basically used to load the preprocessed files
    Methods :-
    1) fit :- This method takes the dataset and masks and fits those to the UNET model
    2) transform :- This method takes the testing files and predicts the output masks
    '''

    def __init__(self,imageLoc,maskLoc,testimageLoc):
        self.imageLoc=imageLoc
        self.maskLoc=maskLoc;
        self.testimageLoc=testimageLoc


    def fit(self):

        images = np.load(self.imageLoc)
        masks = np.load(self.maskLoc)
        train = images[..., np.newaxis]
        masks = masks[..., np.newaxis]
        train = train.astype('float32')
        mask_images = masks.astype('float32')
        model = unet()
        model.fit(train, mask_images, batch_size=8, epochs=10, verbose=1, shuffle=True)
        model.save("unet.h5")


    # predict method to use saved model to predict mask of test images
    def predict(self):
        images = np.load(self.testimageLoc)
        test = images[..., np.newaxis]
        test = test.astype('float32')
        model = unet()
        model.load_weights('unet.h5')
        predicted = model.predict(test, verbose=1)
        np.save('output.npy', predicted)


if __name__ == '__main__':

     parser = argparse.ArgumentParser(description='paths')
     parser.add_argument('--train_path', type=str, help='path for training file')
     parser.add_argument('--test_path', type=str, help='path for testing file')
     parser.add_argument('--mask_path', type=str, help='path for masks')
     args = parser.parse_args()
     model =coburn_unet_model(args.train_path,args.mask_path,args.test_path)
     model.fit()
     #model.predict()






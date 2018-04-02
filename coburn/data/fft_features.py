class Frequency(Transform):
    def __call__(self, images):
        return images.fourier()

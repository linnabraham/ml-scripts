from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
# TODO: This function is already defined in my galactic-rings repo
def get_image_array(img_path, target_size):
    my_image = image.load_img(img_path,target_size=target_size)
    img_array = image.img_to_array(my_image)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale the image manually
    rescaled_img = img_array / 255.0
    return rescaled_img

class trained_model:
    def __init__(self, trained_model_path):
        self.trained_model_path = trained_model_path

    @property
    def model(self):
        return load_model(self.trained_model_path)

    @property
    def image_size(self):
        # Return pretty much every information about your model
        config = self.model.get_config()

        # Return a tuple of width, height and channels as the expected input shape
        batch_input_shape = config["layers"][0]["config"]["batch_input_shape"]
        return batch_input_shape[1:-1]

    def predict_single(self, image_path):
        img_array = get_image_array(img_path=image_path, target_size=self.image_size)
        predictions = self.model.predict(img_array)
        return predictions

    def evaluate(trained_model_path):
        if not os.path.exists(trained_model_path):
            raise ValueError("Invalid model path")

        model = trained_model(trained_model_path).model



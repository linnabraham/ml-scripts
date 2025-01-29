from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import Callback
import numpy as np
# TODO: This function is already defined in my galactic-rings repo
def get_image_array(img_path, target_size):
    my_image = image.load_img(img_path,target_size=target_size)
    img_array = image.img_to_array(my_image)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale the image manually
    rescaled_img = img_array / 255.0
    return rescaled_img

class SaveHistoryCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.history = {'loss': [], 'val_loss': [], 'auc_pr':[], 'val_auc_pr':[], 'val_precision':[], 'val_recall':[]}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['auc_pr'].append(logs.get('auc_pr'))
        self.history['val_auc_pr'].append(logs.get('val_auc_pr'))
        self.history['val_precision'].append(logs.get('val_precision'))
        self.history['val_recall'].append(logs.get('val_recall'))

        with open(self.file_path, 'w') as f:
            json.dump(self.history, f)

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



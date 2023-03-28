from tensorflow import keras

__author__ = "Sebastian Hegenbart"


class YoloModelZoo:
    """"
    This class provides us with a bunch of pre-trained yolo models for
    experimenting and having fun!
    """

    def __init__(self):
        """
        Initializes the object.
        """

        self.available_models = dict()

        # Probably the best model
        #
        self.available_models['yolov1_9M_64x64_7_3_1'] = \
            YoloModelInfo('yolov1_9M_64x64_7_3_1',
                          'yoloapi/data/pre-trained-models/yolov1_9M_64x64_7_3_1.h5',
                          1, 9056280, (64, 64, 3), 7, 3, 1, 512, 500000, 400)

        # Same as above but trained much less
        #
        self.available_models['yolov1_9M_64x64_7_3_2'] = \
            YoloModelInfo('yolov1_9M_64x64_7_3_2',
                          'yoloapi/data/pre-trained-models/yolov1_9M_64x64_7_3_2.h5',
                          1, 9056280, (64, 64, 3), 7, 3, 1, 512, 50000, 40)

        # Smaller model
        #
        self.available_models['yolov1_2.3M_64x64_7_3_1'] = \
            YoloModelInfo('yolov1_2.3M_64x64_7_3_1',
                          'yoloapi/data/pre-trained-models/yolov1_2.3M_64x64_7_3_1.h5',
                          1, 2368320, (64, 64, 3), 7, 3, 1, 512, 500000, 40)

        # Smaller model trained longer.
        #
        self.available_models['yolov1_2.3M_64x64_7_3_2'] = \
            YoloModelInfo('yolov1_2.3M_64x64_7_3_2',
                          'yoloapi/data/pre-trained-models/yolov1_2.3M_64x64_7_3_2.h5',
                          1, 2368320, (64, 64, 3), 7, 3, 1, 512, 500000, 400)

    def list_available_models(self):
        """
        Prints a list of available pre-trained yolo v1 models.
        """
        for key in self.available_models:
            self.available_models[key].print_info()

    def load_model(self, model_id):
        """
        Loads a pre-trained yolo model. Notice, the model is not
        trainable and can only be used for predicting.

        Parameters
        ----------
        model_id : str
            The id of the model to load as returned by list_available_models().

        Returns
        -------
        tuple : (tf.keras.Sequential, YoloModelInfo)
            Returns a tuple containing the pre-trained yolo object detector and
            the corresponding model info object.
        """
        model_info = self.available_models[model_id]

        # Supply a dummy loss function, this can not be trained anymore.
        # Implement your own loss!
        #
        model = keras.models.load_model(model_info.filename, custom_objects={
            'loss': lambda z, zhat: z - zhat})
        return model, model_info


class YoloModelInfo:
    """
    This class is used to hold information about pre-trained yolo models.
    """

    def __init__(self, model_id, filename, yolo_version, trainable_parameters,
                 input_shape, n_cells, n_classes,
                 bb_per_cell, train_batch_size, train_dataset_size,
                 train_epochs):
        self._model_id = model_id
        self.filename = filename
        self.yolo_version = yolo_version
        self.trainable_parameters = trainable_parameters
        self.input_shape = input_shape
        self.n_cells = n_cells
        self.n_classes = n_classes
        self.bb_per_cell = bb_per_cell
        self.train_batch_size = train_batch_size
        self.train_dataset_size = train_dataset_size
        self.train_epochs = train_epochs

    def model_id(self):
        """
        Returns the model's id.

        Returns
        -------
        string
            The model's id is returned.

        """
        return self._model_id

    def print_info(self):
        """
        Prints out a human readable summary of the model.
        """

        print('Model-ID: %s' % self._model_id)
        print('---------------------------------------------')
        print('Input-Shape: {0}'.format(self.input_shape))
        print('n_cells: {0}\nn_classes: {1}\nbb_per_cell: {2}'.format(
            self.n_cells, self.n_classes, self.bb_per_cell))
        print('# Trainable weights and biases: {:,}'.format(
            self.trainable_parameters))
        print('Trained on {:,} inputs'.format(
            self.train_dataset_size * self.train_epochs))
        print('#############################################\n')

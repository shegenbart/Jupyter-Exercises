from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU, Reshape
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

__author__ = "Sebastian Hegenbart"


class YoloModelFactory:
    """
        I took this architecture from
        https://github.com/JY-112553/yolov1-keras-voc/blob/master/models
        /model_tiny_yolov1.py
    """

    @staticmethod
    def create_9m_yolov1_model(input_shape, n_cells, n_classes):
        """
        Creates a yolo object detector model with 9 million parameters.

        Parameters
        ----------
        input_shape : tuple of int
            Defines the shape of the training inputs including color channels
            (e.g.: (64,64,3)).
        n_cells : int
            The number of cells used for predicting bounding boxes (notice,
            that we use the same number in the vertical and horizontal
            direction).
        n_classes : int
            The number of object classes found in the training dataset.

        Returns
        -------
        Sequential
            A keras compatible Sequential yolo v1 model.
        """
        model = Sequential()

        # Layer 1
        model.add(
            Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Layer 2 - 5
        for i in range(0, 4):
            model.add(
                Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same',
                       use_bias=False))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 6
        model.add(
            Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

        # Notice: No max pooling in this layer
        model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                         use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())

        # We do not support more than one bb per cell
        #
        model.add(Dense(units=n_cells * n_cells * (5 + n_classes), name='fc_2'))

        # Reshape such that we can use the GT tensor shape for implementing
        # the loss function. This is
        # easier to understand and also faster.
        #
        # Ignore batch size, we can not now tf keras will figure it out for us.
        #
        model.add(Reshape((n_cells, n_cells, (5 + n_classes))))

        return model

    @staticmethod
    def create_2p3m_yolov1_model(input_shape, n_cells, n_classes):
        """
        Creates a yolo object detector model with 2.3 million parameters. This
        is the same architecture (with fewer filters) as provided by the
        create_9m_yolov1_model() function.

        Parameters
        ----------
        input_shape : tuple of int
            Defines the shape of the training inputs including color
            channels
            (e.g.: (64,64,3)).
        n_cells : int
            The number of cells used for predicting bounding boxes (
            notice,
            that we use the same number in the vertical and horizontal
            direction).
        n_classes : int
            The number of object classes found in the training dataset.

        Returns
        -------
        Sequential
            A keras compatible Sequential yolo v1 model.
        """
        model = Sequential()

        # Layer 1
        model.add(
            Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Layer 2 - 5
        for i in range(0, 4):
            model.add(
                Conv2D(16 * (2 ** i), (3, 3), strides=(1, 1), padding='same',
                       use_bias=False))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 6
        model.add(
            Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

        # Notice: No max pooling in this layer
        model.add(
            Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())

        # We do not support more than one bb per cell
        #
        model.add(Dense(units=n_cells * n_cells * (5 + n_classes), name='fc_2'))

        # Reshape such that we can use the GT tensor shape for implementing the
        # loss function. This is easier to understand and also faster.
        #
        # Ignore batch size, we can not now tf keras will figure it out for us.
        #
        model.add(Reshape((n_cells, n_cells, (5 + n_classes))))

        return model

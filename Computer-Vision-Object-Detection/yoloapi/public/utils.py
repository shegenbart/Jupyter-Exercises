import numpy as np

__author__ = "Sebastian Hegenbart"


class YoloUtils:

    @staticmethod
    def _singletensor_to_bboxes(tensor, img_shape=(64, 64)):
        """
        This method takes an output tensor of a yolo object detector and
        converts it into a list of bounding boxes using image coordinates
        instead of cell-coordinates as predicted by yolo.

        Parameters
        ----------
        tensor : numpy.array - (n_cells, n_cells, 5 + n_classes)
            A yolo prediction tensor.
        img_shape : tuple of ints
            The input shape used for training the object detector in the form
            (rows, columns).

        Returns
        -------
        lists of tuples : [(conf, (x,y,w,h), numpy.array([c0,...,cn])), ...]
            Returns a list of bounding box tuples in image coordinates with
            the confidence value and the predicted one-hot-encoded class
            activations (note to apply softmax to this vector if probabilities
            are required).
        """
        assert len(tensor.shape) == 3, \
            'FAIL: expected numpy.array of shape (n_cells, n_cells, 5 + ' \
            'n_classes)'

        # Figure out number of cells.
        #
        n_cells_h = tensor.shape[0]
        n_cells_w = tensor.shape[1]
        n_classes = tensor.shape[2] - 5  # XXX: works only for 1 bb per cell

        # We create a tensor (cell_offset) with zeros and replace all entries
        # for the X,Y coordinate of bounding boxes with the cell image
        # coordinates of the upper left position.
        #
        # We then create a tensor (cell_scale) with ones and replace all
        # entries for the X,Y coordinate with the cell_width, cell_scale.
        #
        # Finally we compute the correct image coordinates for all predictors
        # in a batch with a simple tensor operation:
        # cell_offset + tensor * cell_scale.
        # Which will transform all cell-coordinates into image coordinates in
        # one tensor operation. This operation will set some X,Y values to
        # non-zero values but does not touch the confidence, so we do not care!
        #

        # predictors are (conf, x, y, w, h, class_id)
        # x,y are the top left in cell coordinates (0,0) is the top left of the
        # cell, (1,1) is the bottom right of the cell w,h are scaled in
        # cell width and height (1,1) = (cell_width, cell_height)
        #
        cell_h_scale = img_shape[0] / n_cells_h  # rows -> height
        cell_w_scale = img_shape[1] / n_cells_w  # cols -> width

        cell_offset = np.zeros((n_cells_h, n_cells_w, 5 + n_classes))
        cell_scale = np.ones((n_cells_h, n_cells_w, 5 + n_classes))

        x = np.arange(0, img_shape[0], img_shape[0] / n_cells_h)
        y = np.arange(0, img_shape[1], img_shape[1] / n_cells_w)
        xi, yi = np.meshgrid(x, y)

        for i in range(len(xi)):
            for j in range(len(yi)):
                # Notice how we have to switch x,y for row,col
                #

                # Set the x-image coordinate values in the position of the
                # bounding box x-coordinate
                #
                cell_offset[j, i, 1] = xi[i, j]

                # Set the y-image coordinate values in the position of the
                # bounding box y-coordinate
                #
                cell_offset[j, i, 2] = yi[i, j]
                cell_scale[j, i, 1] = cell_w_scale
                cell_scale[j, i, 2] = cell_h_scale

        # print(cell_offset)

        # This will scale the x,y values into image coordinates. BUT: YOLO
        # predicts the center coordinates not the upper left we still have to
        # correct for this one.
        #
        image_coord_tensor = cell_offset + tensor * cell_scale

        # This tensor is used to scale the width and height into image
        # coordinates.
        #
        scale_tensor = np.ones((tensor.shape))
        scale_tensor[:, :, 3:5] = (cell_w_scale, cell_h_scale)

        image_coord_tensor = image_coord_tensor * scale_tensor

        # Finally correct for top left instead of center.
        #
        # noinspection PyRedundantParentheses
        top_left_correction = np.zeros((image_coord_tensor.shape))
        top_left_correction[:, :, 1:3] = image_coord_tensor[:, :, 3:5] * 0.5
        image_coord_tensor = image_coord_tensor - top_left_correction

        predictors = image_coord_tensor.reshape(-1, image_coord_tensor.shape[2])

        ret = list()
        for i in range(predictors.shape[0]):
            tmp = (predictors[i, 0], (predictors[i, 1], predictors[i, 2],
                                      predictors[i, 3], predictors[i, 4]),
                   (predictors[i, 5:]))
            ret.append(tmp)
        return ret

    @staticmethod
    def _batchtensor_to_bboxes(tensor, img_shape=(64, 64)):
        """
        This method takes a batch of output tensors of a yolo object detector
        and converts it into a list of lists of bounding boxes using image
        coordinates instead of cell-coordinates as predicted by yolo.

        Parameters
        ----------
        tensor : tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
           A yolo prediction tensor.
        img_shape : tuple of ints
           The input shape used for training the object detector in
           the form
           (rows, columns).

        Returns
        -------
        lists of tuples : [(conf, (x,y,w,h), numpy.array([c0,...,cn])), ...]
           Returns a list of bounding box tuples in image coordinates
           with
           the confidence value and the predicted one-hot-encoded class
           activations (note to apply softmax to this vector if
           probabilities are required).
        """
        assert len(tensor.shape) == 4, \
            'FAIL: expected tensor of shape ' \
            '(batch_size, n_cells, n_cells, 5 + n_classes)'

        # Figure out number of cells.
        #
        n_cells_h = tensor.shape[1]
        n_cells_w = tensor.shape[2]
        n_classes = tensor.shape[3] - 5  # XXX: works only for 1 bb per cell

        # We create a tensor (cell_offset) with zeros and replace all entries
        # for the X,Y coordinate of bounding boxes with the cell image
        # coordinates of the upper left position.
        #
        # We then create a tensor (cell_scale) with ones and replace all
        # entries for the X,Y coordinate with the cell_width, cell_scale.
        #
        # Finally we compute the correct image coordinates for all predictors
        # in a batch with a simple tensor operation:
        # cell_offset + tensor * cell_scale. Which will transform all
        # cell-coordinates into image coordinates in one tensor operation. This
        # operation will set some X,Y values to non-zero values but does not
        # touch the confidence, so we do not care!
        #

        # predictors are (conf, x, y, w, h, class_id)
        # x,y are the top left in cell coordinates (0,0) is the top left of the
        # cell, (1,1) is the bottom right of the cell w,h are scaled in cell
        # width and height (1,1) = (cell_width, cell_height)
        #
        cell_h_scale = img_shape[0] / n_cells_h  # rows -> height
        cell_w_scale = img_shape[1] / n_cells_w  # cols -> width

        cell_offset = np.zeros((n_cells_h, n_cells_w, 5 + n_classes))
        cell_scale = np.ones((n_cells_h, n_cells_w, 5 + n_classes))

        x = np.arange(0, img_shape[0], img_shape[0] / n_cells_h)
        y = np.arange(0, img_shape[1], img_shape[1] / n_cells_w)
        xi, yi = np.meshgrid(x, y)

        for i in range(len(xi)):
            for j in range(len(yi)):
                # Notice how we have to switch x,y for row,col
                #
                # Set the x-image coordinate values in the position of the
                # bounding box x-coordinate
                #
                cell_offset[j, i, 1] = xi[i, j]
                # Set the y-image coordinate values in the position of the
                # bounding box y-coordinate
                #
                cell_offset[j, i, 2] = yi[i, j]
                cell_scale[j, i, 1] = cell_w_scale
                cell_scale[j, i, 2] = cell_h_scale

        # print(cell_offset)

        # This will scale the x,y values into image coordinates. BUT: YOLO
        # predicts the center coordinates not the upper left we still have to
        # correct for this one.
        #
        image_coord_tensor = cell_offset + tensor * cell_scale

        # This tensor is used to scale the width and height into image
        # coordinates.
        #
        scale_tensor = np.ones((tensor.shape))
        scale_tensor[:, :, :, 3:5] = (cell_w_scale, cell_h_scale)

        image_coord_tensor = image_coord_tensor * scale_tensor

        # Finally correct for top left instead of center.
        #
        top_left_correction = np.zeros((image_coord_tensor.shape))
        top_left_correction[:, :, :, 1:3] = \
            image_coord_tensor[:, :, :, 3:5] * 0.5
        image_coord_tensor = image_coord_tensor - top_left_correction

        all_ret = []
        for img_idx in range(image_coord_tensor.shape[0]):

            tensor = image_coord_tensor[img_idx, :]
            predictors = tensor.reshape(-1, image_coord_tensor.shape[3])

            ret = list()
            for i in range(predictors.shape[0]):
                tmp = (predictors[i, 0], (predictors[i, 1], predictors[i, 2],
                                          predictors[i, 3], predictors[i, 4]),
                       (predictors[i, 5:]))
                ret.append(tmp)

            all_ret.append(ret)

        return all_ret

    @staticmethod
    def convert_yolotensor_to_bboxes(inp, img_shape=(64, 64)):
        """
        Converts a yolo object prediction tensor or a batch of yolo object
        prediction tensors into a list of bounding boxes using image-coordinates
        instead of cell-coordinates.

        Parameters
        ----------
        inp : numpy.array - (n_cells, n_cells, 5 + n_classes)
            Object prediction tensor as used by yolo for a single image.
        inp : numpy.array - (batch_size, n_cells, n_cells, 5 + n_classes)
            Object prediction tensor as used by yolo for a batch of images.
        img_shape : tuple of int
            Shape of the input used for training the object detector
            i.e. (rows, columns).

        Returns
        -------
        list of list of tuples
            If the input is 4-dimensional (a prediction of an entire batch)
            a list of lists of bounding boxes is returned,
            i.e.: [ [(conf, (x,y,w,h), numpy.array([c0,...,cn])), ...], ...].
        list of tuples
            If the input is 3-dimensional (a prediction of a single input)
            a list of bounding boxes is returned,
            i.e.: [(conf, (x,y,w,h), numpy.array([c0,...,cn])), ...].
        """
        # Check dimension of inp and use batch or single version.
        #
        if len(inp.shape) == 4:
            return YoloUtils._batchtensor_to_bboxes(inp, img_shape)
        elif len(inp.shape) == 3:
            return YoloUtils._singletensor_to_bboxes(inp, img_shape)
        else:
            print('FAIL: Input shape is invalid, expected 4-dimensions for '
                  'batch, 3-dimensions for single tensor inputs')

    @staticmethod
    def _detect_objects_batch(model, image_batch):
        """
        This method is a wrapper for running a prediction on a full batch
        of input images. The method will normalize the input and then predict
        a yolo tensor for the entire batch.

        Parameters
        ----------
        model : tf.keras.Sequential
            A trained yolo object detector.
        image_batch : numpy.array - (batch_size, height, width, color-channels)
            A batch of images with potential objects to predict.

        Returns
        -------
            numpy.array - (batch_size, n_cells, n_cells, 5 + n_classes)
            Output of the yolo Sequential model for the entire batch.
        """
        x = np.zeros((len(image_batch), *image_batch[0].shape))
        for idx, img in enumerate(image_batch):
            # Normalize image
            #
            image_norm = YoloUtils.normalize(img)
            x[idx, :] = image_norm

        # Predict objects
        #
        yhat = model.predict(x)
        return yhat

    @staticmethod
    def _detect_objects_single(model, image):
        """
        This method is a wrapper for running a prediction on a single input
        image. The method will normalize the input and then predict
        a yolo tensor for a batch of size 1. The function will then extract
        the dimension of the image from the output tensor.

        Parameters
        ----------
        model : tf.keras.Sequential
            A trained yolo object detector.
        image : numpy.array - (height, width, color-channels)
            An image to detect objects in.
        Returns
        -------
        numpy.array - (n_cells, n_cells, 5 + n_classes)
            Output of the yolo Sequential model for a single image.
        """
        # Normalize image
        #
        image_norm = YoloUtils.normalize(image)

        # Predict objects
        #
        yhat = model.predict(np.array([image_norm]))
        return yhat[0, :]

    @staticmethod
    def detect_objects(model, inp):
        """
        This method is used to predict bounding boxes for either a batch
        of inputs or a single input. The method will normalize all inputs
        according to the normalization used in yolo v1. The method is a wrapper
        for either _detect_objects_single() if the input is 3-dimensional or
        _detect_objects_batch() if the input is 4-dimensional.

        Parameters
        ----------
        model : tf.keras.Sequential
            A trained yolo object detector.
        inp : numpy.array(height, width, color-channels)
            A single image for predicting objects.
        inp : numpy.array(batch-size, height, width, color-channels)
            A batch of images for predicting objects.
        Returns
        -------
        numpy.array - (n_cells, n_cells, 5 + n_classes)
            If the input was a single image a single yolo predictor is returned.
        numpy.array - (batch_size, n_cells, n_cells, 5 + n_classes)
            If the input was a batch of images, a batch of yolo predictors
            are returned.
        """
        if type(inp) is list:
            return YoloUtils._detect_objects_batch(model, inp)
        else:
            return YoloUtils._detect_objects_single(model, inp)

    @staticmethod
    def softmax(x):
        """
        Computes the softmax function for each entry of the input vector.
        \\[
            \\hat{x_i} = \\frac{e^{x_i}}{\\sum_{j=1}^{n} e^{x_j}}
        \\]

        Parameters
        ----------
        x : numpy-array - (n)
            A 1-dimensional vector of class activations.
        Returns
        -------
        numpy.array - (n)
            Returns a vector of softmax values of the input, i.e.:
            $$[\\hat{x_1}, ..., \\hat{x_n}]$$
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def normalize(x):
        """
        Normalizes the input to zero mean and unit variance.

        Parameters
        ----------
        x : numpy.array - (rows, cols, channels)
            A numpy array representing an image.

        Returns
        -------
        numpy.array - (rows, cols, channels)
            A numpy array of normalized image values.
        """
        y = (x - np.mean(x)) / (np.std(x) + 0.0001)
        return y

    @staticmethod
    def convert_from_cell_to_img_coords(tensor, img_shape):
        """
        Convert a batch of yolo prediction tensors from using cell-coordinates
        to using image-coordinates for the values of x,y,w and h.

        Parameters
        ----------
        tensor : numpy.array - (batch_size, n_cells, n_cells, 5 + n_classes)
            A batch of yolo prediction tensors.
        img_shape : tuple of int
            The input shape used for training the object detector
            (i.e.: rows, columns).

        Returns
        -------
        numpy.array - (batch_size, n_cells, n_cells, 5 + n_classes)
            A batch of yolo prediction tensors converted from cell-coordinates
            to image-coordinates.

        """
        # Figure out number of cells.
        #
        n_cells_h = tensor.shape[1]
        n_cells_w = tensor.shape[2]
        n_classes = tensor.shape[3] - 5  # XXX: works only for 1 bb per cell

        # We create a tensor (cell_offset) with zeros and replace all entries
        # for the X,Y coordinate of bounding boxes with the cell image
        # coordinates of the upper left position.
        #
        # We then create a tensor (cell_scale) with ones and replace all
        # entries for the X,Y coordinate with the cell_width, cell_scale.
        #
        # Finally we compute the correct image coordinates for all predictors
        # in a batch with a simple tensor operation:
        # cell_offset + tensor * cell_scale. Which will transform all
        # cell-coordinates into image coordinates in one tensor operation. This
        # operation will set some X,Y values to non-zero values but does not
        # touch the confidence, so we do not care!
        #

        # predictors are (conf, x, y, w, h, class_id)
        # x,y are the top left in cell coordinates (0,0) is the top left of the
        # cell, (1,1) is the bottom right of the cell w,h are scaled in cell
        # width and height (1,1) = (cell_width, cell_height)
        #
        cell_h_scale = img_shape[0] / n_cells_h  # rows -> height
        cell_w_scale = img_shape[1] / n_cells_w  # cols -> width

        cell_offset = np.zeros((n_cells_h, n_cells_w, 5 + n_classes))
        cell_scale = np.ones((n_cells_h, n_cells_w, 5 + n_classes))

        x = np.arange(0, img_shape[0], img_shape[0] / n_cells_h)
        y = np.arange(0, img_shape[1], img_shape[1] / n_cells_w)
        xi, yi = np.meshgrid(x, y)

        for i in range(len(xi)):
            for j in range(len(yi)):
                # Notice how we have to switch x,y for row,col
                #

                # Set the x-image coordinate values in the position of the
                # bounding box x-coordinate
                #
                cell_offset[j, i, 1] = xi[i, j]
                # Set the y-image coordinate values in the position of the
                # bounding box y-coordinate
                #
                cell_offset[j, i, 2] = yi[i, j]
                cell_scale[j, i, 1] = cell_w_scale
                cell_scale[j, i, 2] = cell_h_scale

        # This will scale the x,y values into image coordinates. BUT: YOLO
        # predicts the center coordinates not the upper left
        # we still have to correct for this one.
        #
        image_coord_tensor = cell_offset + tensor * cell_scale

        # This tensor is used to scale the width and height into image
        # coordinates.
        #
        scale_tensor = np.ones((tensor.shape))
        scale_tensor[:, :, :, 3:5] = (cell_w_scale, cell_h_scale)

        image_coord_tensor = image_coord_tensor * scale_tensor

        # Finally correct for top left instead of center.
        #
        top_left_correction = np.zeros((image_coord_tensor.shape))
        top_left_correction[:, :, :, 1:3] = \
            image_coord_tensor[:, :, :, 3:5] * 0.5
        image_coord_tensor = image_coord_tensor - top_left_correction

        return image_coord_tensor

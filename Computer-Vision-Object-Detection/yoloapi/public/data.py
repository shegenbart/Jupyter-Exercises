import random
import time

import cv2
import numpy as np
import tensorflow
from sklearn import preprocessing

from yoloapi.public.utils import YoloUtils

__author__ = "Sebastian Hegenbart"


class ToyDatasetGenerator:
    """
    We use this class to create toy data-sets for training object detectors.
    The class is capable of generating random images with a number of
    geometrical objects.
    """

    def __init__(self, rows, columns):
        """
        Initializes the object.

        Parameters
        ----------
        rows : int
            The number of rows of the generated data (height).
        columns : int
            The number of columns of the generated data (height).
        """

        self.img_shape = (rows, columns, 3)  # We always create color images
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.n_classes = 3
        self._init_label_encoder()

    def _init_label_encoder(self):
        """
        Initializes the label encoder.
        """

        # The labels are one-hot-encoded this means
        # 1 = 1,0,0,0
        # 2 = 0,1,0,0
        # and so on, for this purpose we use a LabelBinarizer from sklearn.
        #
        self.label_encoder = preprocessing.LabelBinarizer()
        self.label_encoder.fit_transform(range(0, self.n_classes))

    def generate_data(self, min_objects, max_objects, n_images,
                      min_noise_level=1, max_noise_level=64):
        """
        Generates a batch of random images containing a number of
        objects and the corresponding ground-truth that can be used to train
        and test object detectors in a reasonable time-frame.

        Parameters
        ----------
        min_objects : int
            Minimum number of objects in each image, must be >= 1 and
            <= max_objects.
        max_objects : int
            Maximum number of objects in each image, must be >= minObjects.
            (Notice that a high number of objects can lead to scenarios where
            the objects can not be fitted meaningfully into the parent image.
            The algorithm will try to fit in objects until a maximum number of
            steps and then give up on this image and restart. In such cases
            it is likely that less images with max_objects are generated.
        n_images : int
            Number of images to generate.
        min_noise_level : int, default=1
            The minimum level of noise added to the image, must be >= 1 and
            < max_noise_level.
        max_noise_level : int, default=64
            The maximum noise level added to the image, must be > minNoiseLevel
            and < 255.

        Returns
        -------
        dictionary : Keys - {'X', 'Y'}

        dictionary['X']: list of numpy.array - (rows, cols, channels)
            Each of the generated images (num_images) is stored in a list
            of numpy.arrays which represent the image.

        dictionary['Y']: list of list of bounding boxes
            The corresponding ground-truth of bounding boxes is stored
            as a list of lists of tuples, i.e.:
            [ [(x,y,w,h), numpy.array([c0,...,c1])), ...], ...].
            Each tuple represents a single bounding box and the one-hot-encoded
            class label. By convention we use as class-labels:
                rectangle = 0 : [1,0,0],
                circle = 1 : [0,1,0],
                triangle = 2 : [0,0,1].

            Notice that the indices of dictionary['X'] and dictionary['Y']
            correspond such that dictionary['Y'][idx] will provide all bounding
            boxes of the image stored in dictionary['X'][idx].

        """

        assert 1 <= min_objects <= max_objects, \
            "minObjects must be >= 1, maxObjects must be >= minObjects"

        assert 0 <= min_noise_level < max_noise_level <= 255, \
            "minNoiseLevel must be >= 0 and < 255 and maxNoiseLevel " \
            "must be > minNoiseLevel"

        data = dict()
        data['X'] = []
        data['Y'] = []
        start_time = time.time()
        while len(data['X']) < n_images:
            # noinspection PyBroadException
            try:
                img, gt = self._gen_image(self.img_shape, min_objects,
                                          max_objects, self.colors,
                                          min_noise_level,
                                          max_noise_level)
                data['X'].append(img)
                data['Y'].append(gt)

                if len(data['X']) % 100 == 0:
                    print('.', end='')
                if len(data['X']) % 5000 == 0:
                    print(' %d' % len(data['X']))
            except:
                pass

        end_time = time.time()

        print('Generated %d images in %.2f seconds.' % (
            len(data['X']), end_time - start_time))
        return data

    @staticmethod
    def draw_bbox(img, bbox):
        """
        Draws a bounding box into img, modifying the underlying data of img.

        Parameters
        ----------
        img : numpy.array - (rows, columns, 3)
            An image represented by a numpy.array, assumes 3 color channels.
        bbox : tuple of floats
            A bounding box tuple (x,y,w,h).
        """
        cv2.rectangle(img, (bbox[0], bbox[1]),
                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 0), 1)

    def _gen_rect(self, img_dims, posx, posy, width, height, angle, color):
        """
        Generates a rectangular-shaped object given the input parameters. This
        function also determines the bounding box for the generated object.

        Parameters
        ----------
        img_dims : tuple of int
            The dimensions of the parent image this object will be embedded in.
        posx :  int
         The horizontal position of the upper left corner of the rectangle
         before rotation.
        posy : int
            The vertical position of the upper left corner of the rectangle
            before rotation.
        width : int
            The width of the rectangle.
        height : int
            The height of the rectangle.
        angle : float
            The rotation angle in degrees applied to the rectangle.
        color : triplet of int
            The RBG color to fill the object.
        Returns
        -------
        tuple : (numpy.array, tuple)
            Returns a tuple (patch, bbox) where patch contains an image with
            the embedded object (all non-object pixels will be 0), bbox is a
            tuple representing bounding box of the object (x,y,w,h).

        """

        patch = np.zeros(img_dims, dtype='uint8')

        patch = cv2.rectangle(patch, (posx, posy),
                              (posx + width, posy + height), color, -1)
        rot_matrix = cv2.getRotationMatrix2D(
            (posx + int(width / 2), posy + int(height / 2)), angle, 1.0)
        patch = cv2.warpAffine(patch, rot_matrix, patch.shape[1::-1],
                               flags=cv2.INTER_LINEAR)

        bbox = self._detect_bbox(patch)
        return patch, bbox

    def _gen_circle(self, img_dims, posx, posy, radius, color):
        """
        Generates a circle-shaped object given the input parameters. This
        function also determines the bounding box for the generated object.

        Parameters
        ----------
        img_dims : tuple of int
            The dimensions of the parent image this object will be embedded in
            (rows, columns).
        posx : int
            The horizontal position of the center of the circle.
        posy : int
            The vertical position of the center of the circle.
        radius : float
            The radius of the circle.
        color : triplet
            RBG color to fill the rectangle.
        Returns
        -------
        tuple : (numpy.array, tuple)
            Returns a tuple (patch, bbox) where patch contains an image with
            the embedded object (all non-object pixels will be 0), bbox is a
            tuple representing bounding box of the object (x,y,w,h).
        """

        patch = np.zeros(img_dims, dtype='uint8')
        patch = cv2.circle(patch, (posx, posy), radius, color, -1)
        bbox = self._detect_bbox(patch)

        return patch, bbox

    def _gen_triangle(self, img_dims, posx, posy, side_len, angle, color):
        """
        Generates an equilateral triangle-shaped object given the input
        parameters. This function also determines the bounding box for the
        generated object.

        Parameters
        ----------
        img_dims : tuple of int
            The dimensions of the parent image this object will be embedded in
            (rows, columns).
        posx : int
            The horizontal position of the center-tip of the triangle in
            standard orientation.
        posy : int
            The horizontal position of the center-tip of the triangle in
            standard orientation.
        side_len: int
            The side-length of the triangle.
        angle : float
            The rotation angle in degrees applied to the rectangle.
        color : triplet of int
            The RBG color to fill the object.

        Returns
        -------
        tuple : (numpy.array, tuple)
            Returns a tuple (patch, bbox) where patch contains an image with
            the embedded object (all non-object pixels will be 0), bbox is a
            tuple representing bounding box of the object (x,y,w,h).
        """
        patch = np.zeros(img_dims, dtype='uint8')

        pt1 = (posx, posy)
        pt2 = (posx - side_len / 2, posy - side_len)
        pt3 = (posx + side_len / 2, posy - side_len)

        cv2.fillConvexPoly(patch, np.array([[pt1, pt2, pt3]], dtype='int32'),
                           color)

        rot_matrix = cv2.getRotationMatrix2D(((posx), posy + int(side_len / 2)),
                                             angle, 1.0)
        patch = cv2.warpAffine(patch, rot_matrix, patch.shape[1::-1],
                               flags=cv2.INTER_LINEAR)

        bbox = self._detect_bbox(patch)
        return patch, bbox

    @staticmethod
    def _detect_bbox(img):
        """
        This function is used to determine the bounding box of the object. We
        always rotate along the center of the object, not the image, we
        therefore simply re-detect the object.

        Parameters
        ----------
        img : numpy.array - (rows, columns, channels)
            An input image containing a single object.

        Returns
        -------
        tuple of float
            Returns a tuple representing the bounding box of the object inside
            img as (x,y,w,h).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = np.max(gray)
        ret, thresh = cv2.threshold(gray, 0, threshold, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[-2:]
        bbox = cv2.boundingRect(contours[0])
        return bbox

    @staticmethod
    def _is_inside(bbox, pt):
        """
        Checks if a point is within a bounding box.

        Parameters
        ----------
        bbox : tuple of floats
            A tuple representing a bounding box (x,y,w,h).
        pt : tuple of floats
            A tuple representing a 2-dimensional point (x,y).

        Returns
        -------
        boolean
            True if pt is within bbox, False otherwise.
        """

        bbox_x1 = bbox[0]
        bbox_x2 = bbox[0] + bbox[2]
        bbox_y1 = bbox[1]
        bbox_y2 = bbox[1] + bbox[3]

        x = pt[0]
        y = pt[1]

        if (bbox_x1 <= x <= bbox_x2 and bbox_y1 <= y <= bbox_y2):
            return True
        return False

    def _bbox_intersect(self, bbox1, bbox2):
        """
        This method checks if bounding boxes intersect, this means that
        generated objects are overlapping. We do not want to generate such
        images.

        Parameters
        ----------
        bbox1 : tuple of floats
            A tuple representing a bounding box (x,y,w,h).
        bbox2 : tuple of floats
            A tuple representing a bounding box (x,y,w,h).

        Returns
        -------
        boolean
            True if bbox1 and bbox2 have a non-zero intersection, False
            otherwise.
        """

        # Check if any corner of bbox1 is in bbox2 and vice versa.
        #
        box1_pt1 = (bbox1[0], bbox1[1])
        box1_pt2 = (bbox1[0] + bbox1[2], bbox1[1])
        box1_pt3 = (bbox1[0], bbox1[1] + bbox1[3])
        box1_pt4 = (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])

        box2_pt1 = (bbox2[0], bbox2[1])
        box2_pt2 = (bbox2[0] + bbox2[2], bbox2[1])
        box2_pt3 = (bbox2[0], bbox2[1] + bbox2[3])
        box2_pt4 = (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])

        if self._is_inside(bbox1, box2_pt1):
            return True

        if self._is_inside(bbox1, box2_pt2):
            return True

        if self._is_inside(bbox1, box2_pt3):
            return True

        if self._is_inside(bbox1, box2_pt4):
            return True

        if self._is_inside(bbox2, box1_pt1):
            return True

        if self._is_inside(bbox2, box1_pt2):
            return True

        if self._is_inside(bbox2, box1_pt3):
            return True

        if self._is_inside(bbox2, box1_pt4):
            return True

        return False

    @staticmethod
    def _is_bbox_inside_img(img_rows, img_cols, bbox):
        """
        This method determines if a bounding box is fully within an image.

        Parameters
        ----------
        img_rows : int
            The number of image rows (height).
        img_cols : int
            the number of image columns (width).
        bbox : tuple of floats
            A tuple representing a bounding box (x,y,w,h).
        Returns
        -------
        boolean
            True if bbox is within the image, False otherwise.
        """

        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] >= img_cols or \
                bbox[1] + bbox[3] >= img_rows:
            return False

        # Check bbox is square, else it was cut off and can not be inside the
        # image. There is a very small chance that the object was cut such that
        # the bb is a square. We do not care about this anomaly.
        #
        if bbox[2] != bbox[3]:
            return False
        return True

    def _gen_image(self, img_shape, min_shapes, max_shapes, colors_list,
                   min_noise_level, max_noise_level):
        """
        Generate an entire image with a set of random objects.

        Parameters
        ----------
        img_shape : tuple of int
            A tuple representing the shape of the image (rows, columns).
        min_shapes : int
            The minimum number of generated shapes.
        max_shapes : int
            The maximum number of generated shapes.
        colors_list : list of triplets
            A list of possible colors encoded as RGB, i.e. [ (255,0,0), ...,
            (128,128,12) ].
        min_noise_level : int
            The minimum noise level added to the image.
        max_noise_level : int
            The maximum noise level added to the image.

        Returns
        -------
        tuple - (numpy.array, list)
            Returns a tuple (img, gt), where img is a numpy array representing
            the image, gt is a list of tuples. Each tuple represents a bounding
            box with the corresponding one-hot-encoded class label of the
            object, i.e.: [ ((x,y,w,h), np.array([c0,...,cn])), ... ].
        """

        shape = random.randint(0, 2)
        color_idx = random.randint(0, len(colors_list) - 1)
        color = colors_list[color_idx]

        # Create data used during loop
        background = np.zeros(img_shape, dtype='uint8')
        groundtruth = []

        num_shapes = random.randint(min_shapes, max_shapes)
        min_side = min(img_shape[0], img_shape[1])

        shape_side_low = int(min_side / 10)
        shape_side_high = int(min_side / 4)

        shape_cnt = 0
        tries = 0
        while shape_cnt < num_shapes:
            tries = tries + 1
            if tries > 2 * max_shapes:
                raise Exception()

            try:
                if shape == 0:
                    side_len = random.randint(shape_side_low, shape_side_high)
                    rotation = random.randint(0, 180)
                    posx = random.randint(side_len, img_shape[0] - (side_len))
                    posy = random.randint(side_len, img_shape[1] - (side_len))
                    candidate, cand_bbox = self._gen_rect(img_shape, posx,
                                                          posy, side_len,
                                                          side_len,
                                                          rotation, color)

                if shape == 1:
                    side_len = random.randint(shape_side_low, shape_side_high)
                    posx = random.randint(side_len, img_shape[0] - (side_len))
                    posy = random.randint(side_len, img_shape[1] - (side_len))
                    candidate, cand_bbox = self._gen_circle(img_shape, posx,
                                                            posy, side_len,
                                                            color)

                if shape == 2:
                    side_len = random.randint(shape_side_low, shape_side_high)
                    rotation = random.randint(0, 360)
                    posx = random.randint(side_len, img_shape[0] - (side_len))
                    posy = random.randint(side_len, img_shape[1] - (side_len))
                    candidate, cand_bbox = self._gen_triangle(img_shape,
                                                              posx, posy,
                                                              side_len,
                                                              rotation,
                                                              color)

                if not self._is_bbox_inside_img(img_shape[0], img_shape[1],
                                                cand_bbox):
                    continue

                # Check we have no overlap between the new shape and any
                # previous shape
                #
                intersect = False
                for bbox, label in groundtruth:

                    if self._bbox_intersect(bbox, cand_bbox):
                        intersect = True
                        break

                # If we intersect try again.
                #
                if intersect:
                    continue

                background = background + candidate
                groundtruth.append(
                    (cand_bbox, self.label_encoder.transform([shape])[0]))
                shape_cnt = shape_cnt + 1

                shape = random.randint(0, 2)
                color_idx = random.randint(0, len(colors_list) - 1)
                color = colors_list[color_idx]
            except:
                pass

        # Add a background and noise
        # We first determine the strength of the noise.
        #
        noise_max_strength = np.random.randint(min_noise_level, max_noise_level)
        if noise_max_strength == 0:
            noise_max_strength = 1

        noise = np.random.randint(0, noise_max_strength, size=img_shape)

        background = background + noise
        background[background > 255] = 255
        return background.astype('uint8'), groundtruth


class YoloBatchGenerator(tensorflow.keras.utils.Sequence):
    """
    This class generates batch data for training the yolo NN. Its most important
    job is to convert the groundtruth to a yolo-compatible format. The class
    is based on tf Sequence and can therefore directly be used with the
    keras.fit methods.
    """
    def __init__(self, training_data, n_cells=7, bb_per_cell=1, n_classes=3,
                 batch_size=512, gen_yolo_y_custom_func = None):
        """
        Initializes the object.

        Parameters
        ----------
        training_data : dictionary - Keys {'X', 'Y'}
            A dictionary containing images and ground-truth object labels as
            generated by the ToyDatasetGenerator class.
        n_cells: int, default=7
            Number of cells (used horizontally and vertically) used to predict
            bounding boxes.
        bb_per_cell : 1, default=1
            Number of cells predicted by each cell, notice: currently we only
            support 1.
        n_classes : int, default=3
            Number of different object types.
        batch_size : int, default=512
            the batch size used for training the object detector (decrease this
            if you are running out of GPU memory).
        """

        assert 'X' in training_data and 'Y' in training_data,  \
            "training_data is of the wrong type, expecting a dictionary" \
            "as returned by the ToyDatasetGenerator class"

        assert bb_per_cell == 1, \
            "currently we support only 1 bb prediction per cell"

        self.training_data = training_data
        self.input_shape = self.training_data['X'][0].shape
        self.n_cells = n_cells
        self.bb_per_cell = bb_per_cell
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.indices = np.arange(len(self.training_data['X']))
        self._gen_yolo_y_custom_func = gen_yolo_y_custom_func

    def _generate_data(self, indices):
        """
        Generates a batch of data for training.

        Parameters
        ----------
        indices : list of int
            Sample-indices for the next batch.

        Returns
        -------
        tuple - (np.array, np.array)
            Returns a tuple (x,y) where x (batch_size, rows, cols, channels) is
            the image data and y (batch_size, n_cells, n_cells, 5 + n_classes)
            is the ground-truth used for training the yolo network.
        """

        # Create data for a batch x : (n_samples, *dim, n_channels)
        #
        x = np.empty((self.batch_size,
                      *self.input_shape))  # we force RGB for now, * unpacks
        # the tuple

        # This is the groundtruth, the shape is #batchsize, cells, cells,
        # bb_per_cell * 5 (x,y,w,h,obj_conf) + #classes
        # Notice that the class is only predicted once per cell, and not per
        # bounding box in yolov1!
        #
        y = np.zeros((self.batch_size, self.n_cells, self.n_cells,
                      (self.bb_per_cell * 5 + self.n_classes)), dtype=np.float32)

        if len(indices) != self.batch_size:
            raise ValueError('Partially empty batch')

        for i, idx in enumerate(indices):
            img = self.training_data['X'][idx]
            gt = self.training_data['Y'][idx]

            # If the input does not match the input_shape we resize it.
            # Notice that we can not resize
            # the 3rd dimension (i.e. color channels). For now we just raise
            # an error.
            # TODO: maybe implement a resize.
            #
            if img.shape != self.input_shape:
                raise ValueError('Data has unexpected dimensions.')

            # We normalize this all the time, could be optimized.
            #
            x[i, ] = YoloUtils.normalize(img)
            if self._gen_yolo_y_custom_func is not None:
                y[i, ] = self._gen_yolo_y_custom_func(gt, self.input_shape[0], self.input_shape[1], self.n_cells, self.n_classes)
            else:
                y[i, ] = self._gen_yolo_y(gt, self.input_shape[0], self.input_shape[1], self.n_cells, self.n_classes)

        return x, y

    # This is probably the most important method here. It creates a correct
    # yolo annotation used for training the neural network.
    # TODO: The students should implement this.
    #
    def _gen_yolo_y(self, gt, img_rows, img_columns, n_cells, n_classes):
        """
        TODO: Leave this to the students.
        The function creates a yolo-compatible groundtruth for a lists
        of bounding boxes.

        Parameters
        ----------
        gt - list of list of tuples
            A list of list of bounding boxes as generated by the
            ToyDatasetGenerator class.

        Returns
        -------
        numpy.array : (batch_size, n_cells, n_cells, 5 + n_classes)
            Returns a numpy array containing yolo-compatible prediction tensors
            used to train the object detector.
        """

        # Remember each cell (self.n_cells * self.n_cells) is capable of
        # predicting self.bb_per_cell bounding
        # boxes. For each prediction there are 5 values (x,y,w,h,confidence)
        # plus there is one single class prediction
        # which is one hot encoded (e.g. 0001 for 4 classes) i.e. self.n_classes
        #
        output_tensor = np.zeros(
            (n_cells, n_cells, self.bb_per_cell * 5 + n_classes),
            dtype=np.float)

        cell_width = int(img_rows / n_cells)
        cell_height = int(img_columns / n_cells)

        # Store the objects associated to the cells using a dictionary,
        # with key (x-index, y-index) of the cell.
        # Values are lists of groundtruth tuples.
        #
        objs_by_cell = dict()

        # For each object in gt we create one bounding box (up to
        # self.bb_per_cell for each cell.)
        #
        for obj_gt in gt:
            # gt is a list of tuples of this form ((x,y,w,h), class) e.g.: ((
            # 25, 62, 26, 26), 2) describing the
            # class and bounding box for each object within the image.
            #
            x, y, w, h = obj_gt[0]

            # For each object, one cell is responsible for predicting the
            # bounding box.
            # NOTE: yolo predicts the CENTER of the bounding box not the
            # upper left corner (such as used in our ground truth.)
            #
            # So which cell is it? It is the cell that contains the CENTER of
            # the bounding box of the object. We therefore have to compute
            # the center of the bounding box and figure out which cell is
            # responsible for predicting this object.
            #
            bbox_center = (x + 0.5 * w, y + 0.5 * h)

            # By assigning a cell to predict the bounding boxes we implicit
            # force the network to train multiple predictors
            # for different areas of the image.
            #
            ass_cell_x = int(bbox_center[
                                 0] / cell_width)  # we start with cell-index
            # 0 therefore we round down
            ass_cell_y = int(bbox_center[
                                 1] / cell_height)  # we start with
            # cell-index 0 therefore we round down

            key = (ass_cell_x, ass_cell_y)
            if key not in objs_by_cell:
                objs_by_cell[key] = []

            objs_by_cell[key].append(obj_gt)

        # For each cell (i.e. each key in the dictionary) check that we have
        # a max of bb_per_cell objects associated.
        # If there are more, we do not have enough space in the output tensor
        # of the neural network and we just discard
        # the last object (this is quite unlikely to happen anyway).
        #
        for cell_key in objs_by_cell:
            n_objs = len(objs_by_cell[cell_key])
            if n_objs > self.bb_per_cell:
                print(
                    'WARNING: More than {0} objects (is {1}) associated with '
                    'cell {2}!\n'.format(
                        self.bb_per_cell, n_objs, cell_key))
                # Remove all objects that do not fit in the output tensor.
                #
                objs_by_cell[cell_key] = objs_by_cell[cell_key][
                                         :self.bb_per_cell]

        # Fill the output tensor with the correct groundtruth.
        #
        for cell_key in objs_by_cell:
            # There is one more thing to do to make it easier for the NN to
            # predict the bounding boxes.
            # Remember NNs do not like to predict in large intervals,
            # we therefore rescale the x,y,w and h
            # values into cell-coordinates. This means:
            # We denote the position of the upper left corner of the
            # associated cell as position (0,0)
            # and the lower right corner of the associated cell as position (
            # 1,1). Remember: We predict the center
            # of the bounding box, and therefore this value will always be in
            # the interval [0,1].
            #

            # Prepare the output tensor for this cell.
            #
            cell_tensor = np.zeros(self.bb_per_cell * 5 + n_classes,
                                   dtype=np.float32)

            # This is the amount of data we have to store for each object
            # associated with this cell.
            #
            len_per_object = 5 + n_classes

            for obj_cnt, obj_gt in enumerate(objs_by_cell[cell_key]):
                x, y, w, h = obj_gt[0]
                class_num = obj_gt[1]
                bbox_center = (x + 0.5 * w, y + 0.5 * h)

                # Upper left position of cell in image coordinates
                #
                x0 = cell_key[0] * cell_width
                y0 = cell_key[1] * cell_height

                # Rescale center position to cell coordinates
                #
                x_scaled = (bbox_center[0] - x0) / cell_width
                y_scaled = (bbox_center[1] - y0) / cell_height

                # Rescale width and height into multiples of cell width and
                # height
                #
                w_scaled = w / cell_width
                h_scaled = h / cell_height

                # Encode the class label (one-hot).
                #
                class_label = class_num

                # Object confidence is 1, it is an object. Notice, we do not
                # fill in any values for all other non-object
                # cells as they are initialized with 0 anyway (this means
                # obj_conf is 0 there). This is respected by the loss
                # function.
                #
                obj_conf = 1

                # Store the data in the output tensor for this cell.
                #
                start = obj_cnt * len_per_object
                end = start + len_per_object

                # NOTICE, WE PUT CONF IN THE 0 INDEX HERE
                # THIS IS HOW OUR LOSS IS IMPLEMENTED BUT I THINK DIVERGES
                # FROM THE PAPER (IT IS INDEX 4 I THINK)
                # FIX THIS MAYBE!
                bbox_data = np.array(
                    [obj_conf, x_scaled, y_scaled, w_scaled, h_scaled])
                cell_tensor[start:end] = np.concatenate(
                    (bbox_data, class_label))

            # Store the entire cell tensor into the output tensor containing
            # all cells.
            #
            output_tensor[cell_key[0], cell_key[1], :] = cell_tensor

        return output_tensor

    def encode_yolo(self, gt):
        """
        Convenience function for the user to test some code. This function
        returns yolo encoded groundtruth for provided list of lists of bounding
        boxes.

        Parameters
        ----------
        gt : list of tuples
            A list of bounding boxes for a single image,
            i.e.: [((x,y,w,h), np.array([c0,..,cn])), ...].

        Returns
        -------
        np.array - (n_cells, n_cells, 5 + n_classes)
            A yolo encoded prediction tensor.
        """
        if self._gen_yolo_y_custom_func is not None:
            y = self._gen_yolo_y_custom_func(gt, self.input_shape[0], self.input_shape[1], self.n_cells, self.n_classes)
        else:
            y = self._gen_yolo_y(gt, self.input_shape[0], self.input_shape[1], self.n_cells, self.n_classes)
        return y

    def on_epoch_end(self):
        """
        Method is called on the end of every epoch by keras. What it does is
        it reshuffles the batch indices for the next epoch.
        """
        self.indices = np.arange(len(self.training_data['X']))
        np.random.shuffle(self.indices)

    # This is the stuff we have to implement for being a Sequence
    #

    def __len__(self):
        """
        Returns the number of batches per epoch.

        Returns
        -------
        int
            The number of batches per epochs (rounded down).
        """

        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """
        Creates the data for the next batch and returns it.

        Parameters
        ----------
        index : int
            The index of the batch to retrieve (i.e.: first batch is 0).

        Returns
        -------
        tuple - (np.array, np.array)

        Returns a tuple (x,y) of tensors where x is the image data and of shape
        (batch_size, rows, cols, channels), y is the ground-truth of shape
        (batch_size, n_cells, n_cells, 5 + n_classes).
        """

        # Create batch index
        #
        indices = self.indices[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # Generate
        x, y = self._generate_data(indices)
        return x, y

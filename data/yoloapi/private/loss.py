import tensorflow as tf

"""
This is my reference implementation of the yolo v1 loss used in Autonome
to supply the loss to keras. Do not give this to the students, they are
supposed to implement it themselves.
"""

__author__ = "Sebastian Hegenbart"


class YoloLossSimple:
    """
    The reference implementation of yolo v1 loss using 1 bounding box
    prediction per cell.
    """

    def __init__(self, n_classes, lambda_noobj=0.5):
        """
        Initializes the class.

        Parameters
        ----------
        n_classes : int
            The number of object-classes used.
        lambda_noobj: float, default=0.5
            The parameter controlling the confidence loss in the range [0,1].
        """
        self.n_classes = n_classes

        assert 0 <= lambda_noobj <= 1, \
            "lambda_noobj is expected to be within the interval [0,1]"
        self.lambda_noobj = lambda_noobj

    def yolo_class_loss_tf(self, z, zhat):
        """
        Computes the yolo class loss, only supports one predicted bounding box
        per cell.

        Parameters
        ----------
        z : tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The ground-truth for computing the class loss.
        zhat: tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The predicted yolo output for computing the class loss.

        Returns
        -------
        float
            The class loss between z and zhat.

        """
        class_idx_start = 5

        # Fetch the green tensor (the class labels)
        #
        z_tf_class_predictions = \
            z[:, :, :, class_idx_start:class_idx_start + self.n_classes]
        zhat_tf_class_predictions = \
            zhat[:, :, :,  class_idx_start:class_idx_start + self.n_classes]

        # Point-wise difference
        #
        tmp = z_tf_class_predictions - zhat_tf_class_predictions

        # Point-wise square
        #
        tmp = tmp * tmp
        # Sum of squared differences over the class labels (i.e. axis = 3),
        # note that axis count starts with 0.
        #
        ssd = tf.reduce_sum(tmp, axis=3)

        z_confidences = z[:, :, :, 0]
        ssd_conf = ssd * z_confidences

        class_loss = tf.reduce_sum(ssd_conf)
        return class_loss

    def yolo_confidence_loss_tf(self, z, zhat):
        """
        Computes the yolo confidence loss, only supports one predicted
        bounding box per cell.

        Parameters
        ----------
        z : tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The ground-truth for computing the class loss.
        zhat: tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The predicted yolo output for computing the class loss.

        Returns
        -------
        float
            The confidence loss between z and zhat.
        """

        # First term
        #
        z_conf = z[:, :, :, 0]
        zhat_conf = zhat[:, :, :, 0]

        diff = z_conf - zhat_conf

        diff_squared = diff * diff
        tmp = z_conf * diff_squared  # We can just reuse Z_conf here

        # This time we sum over all elements, therefore no axis is supplied
        #
        first_term_loss = tf.reduce_sum(tmp)

        # Second term
        #
        noobj_conf = 1 - z[:, :, :, 0]  # Flip 1 to 0 and 0 to 1

        diff = z_conf - zhat_conf
        diff_squared = diff * diff
        tmp = noobj_conf * diff_squared
        second_term_loss = tf.reduce_sum(tmp)

        return first_term_loss + self.lambda_noobj * second_term_loss

    @staticmethod
    def yolo_location_loss_tf(z, zhat):
        """
        Computes the yolo location loss, only supports one predicted
        bounding box per cell.

        Parameters
        ----------
        z : tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The ground-truth for computing the class loss.
        zhat: tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The predicted yolo output for computing the class loss.

        Returns
        -------
        float
            The location loss between z and zhat.
        """

        z_bboxes = z[:, :, :, 1:3]  # Get the x,y values
        zhat_bboxes = zhat[:, :, :, 1:3]  # Get the x,y values

        differences = z_bboxes - zhat_bboxes
        squared_differences = tf.square(differences)

        # Now sum for each bounding box, make sure we only sum the x,y
        # differences for one bounding box.
        #
        squared_sum = tf.reduce_sum(squared_differences, axis=3)

        # Figure out the object confidence
        #
        obj_conf = z[:, :, :, 0]

        tmp = obj_conf * squared_sum
        loss = tf.reduce_sum(tmp)
        return loss

    @staticmethod
    def yolo_shape_loss_tf(z, zhat):
        """
        Computes the yolo shape loss, only supports one predicted
        bounding box per cell.

        Parameters
        ----------
        z : tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The ground-truth for computing the class loss.
        zhat: tf.Tensor - (batch_size, n_cells, n_cells, 5 + n_classes)
            The predicted yolo output for computing the class loss.

        Returns
        -------
        float
            The confidence loss between z and zhat.
        """
        z_shapes = z[:, :, :, 3:5]
        zhat_shapes = zhat[:, :, :, 3:5]

        z_shapes = tf.maximum(z_shapes, 0)
        zhat_shapes = tf.maximum(zhat_shapes, 0)

        # Compute square root
        #
        z_sqroots = tf.sqrt(z_shapes)
        zhat_sqroots = tf.sqrt(zhat_shapes)

        differences = z_sqroots - zhat_sqroots
        squared_differences = differences * differences
        sqd_sum = tf.reduce_sum(squared_differences, axis=3)

        # Fetch confidence
        #
        obj_conf = z[:, :, :, 0]

        tmp = obj_conf * sqd_sum
        loss = tf.reduce_sum(tmp)
        return loss

    def yolo_loss_keras_wrapper(self):
        """
        Keras only takes losses of the form loss(Z,Zhat). To supply loss
        functions that work on more parameters (such as n_classes,
        lambda_noobj) we can simply supply a function that returns a function.
        This is what we do here.

        Returns
        -------
        loss(z, zhat)
            Returns the combined loss function.
        """
        def loss(z, zhat):
            return self.yolo_class_loss_tf(z, zhat) + \
                   self.yolo_confidence_loss_tf(z, zhat) + \
                   self.yolo_location_loss_tf(z, zhat) + \
                   self.yolo_shape_loss_tf(z, zhat)

        return loss

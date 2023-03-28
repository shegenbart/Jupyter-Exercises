import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from yoloapi.public.utils import YoloUtils

__author__ = "Sebastian Hegenbart"


class YoloVisualizer:
    """
    This class provides method for visualizing detected objects.
    """

    def __init__(self, img_shape):
        """
        Initializes the object.

        Parameters
        ----------

        img_shape : tuple of ints
            The shape of the input used to train the object detector, i.e.:
            (rows, columns).
        """
        self.img_shape = img_shape

    def draw_bboxes_tensor(self, img, predicted_tensor, conf_t=-1):
        """
        Draw bounding boxes, class predictions and confidences for all bounding
        boxes with a confidence value >= conf_t. This method does not perform
        non-maximum-suppression.

        Parameters
        ----------
        img : numpy.array - (rows, cols, channels)
            Image used as background, data is not modified.
        predicted_tensor : np.array - (n_cells, n_cells, 5 + n_classes)
            Prediction tensor, output of the yolo model.
        conf_t : float
            Threshold for confidence.
        """
        assert len(predicted_tensor.shape) == 3, \
            'FAIL: expected predicted tensor of shape (n_cells, n_cells, ' \
            '5 + n_classes) but is {0}'.format(predicted_tensor.shape)
        pred_imgc = \
            YoloUtils.convert_from_cell_to_img_coords(
                np.array([predicted_tensor]), self.img_shape).reshape(-1, 8)

        plt.figure(figsize=(7, 7))
        plt.imshow(img)
        ax = plt.gca()

        for idx in range(pred_imgc.shape[0]):
            predictor = pred_imgc[idx]
            conf = predictor[0]

            if conf >= conf_t:

                x, y, w, h = predictor[1], predictor[2], predictor[3], \
                             predictor[4]
                cls = YoloUtils.softmax(predictor[5:])

                rect = Rectangle((x, y), w, h, fill=False, color='#57d977',
                                 linewidth=2)
                ax.add_patch(rect)

                labels = ['Square', 'Circle', 'Triangle', 'Unknown']
                ind = np.argmax(cls)
                label = '%s : (%.2f, %.2f, %.2f), confidence: %.2f' \
                        % (labels[ind], cls[0], cls[1], cls[2], conf)
                t = plt.text(x - 5, y - 5, label, color='white', fontsize=12)
                t.set_bbox(dict(facecolor='#57d977', alpha=0.75,
                                edgecolor='#57d977'))

    @staticmethod
    def draw_bboxes(img, bboxes_list, conf_t=-1):
        """
        Draw bounding boxes, class predictions and confidences for all bounding
        boxes with a confidence value >= conf_t. This method does not perform
        non-maximum-suppression.

        Parameters
        ----------
        img : numpy.array - (rows, columns, channels)
            The image to use as background, data is not modified.
        bboxes_list : lists of tuples
            The list of bounding boxes, i.e.:
            [ (conf, (x,y,w,h), np.array([c0,...,cn])), ...]).
        conf_t : float
            Threshold for confidence.
        """
        plt.figure(figsize=(7, 7))
        plt.imshow(img)
        ax = plt.gca()

        for predictor in bboxes_list:

            conf = predictor[0]
            if conf >= conf_t:

                x, y, w, h = predictor[1]
                cls = YoloUtils.softmax(predictor[2])
                rect = Rectangle((x, y), w, h, fill=False, color='#57d977',
                                 linewidth=2)
                ax.add_patch(rect)

                labels = ['Square', 'Circle', 'Triangle', 'Unknown']
                ind = np.argmax(cls)
                label = '%s : (%.2f, %.2f, %.2f), confidence: %.2f' \
                        % (labels[ind], cls[0], cls[1], cls[2], conf)
                t = plt.text(x - 5, y - 5, label, color='white', fontsize=12)
                t.set_bbox(dict(facecolor='#57d977', alpha=0.75,
                                edgecolor='#57d977'))
                
               
    @staticmethod
    def draw_bboxes_gt(img, gt_bboxes_list):
        """
        Draw bounding boxes using ground-truth data (there is no confidence in ground-truth bounding box data). 
        This method does not perform non-maximum-suppression.

        Parameters
        ----------
        img : numpy.array - (rows, columns, channels)
            The image to use as background, data is not modified.
        bboxes_list : lists of tuples
            The list of bounding boxes, i.e.:
            [ ((x,y,w,h), np.array([c0,...,cn])), ...]).
        conf_t : float
            Threshold for confidence.
        """
        plt.figure(figsize=(7, 7))
        plt.imshow(img)
        ax = plt.gca()

        for predictor in gt_bboxes_list:

            x, y, w, h = predictor[0]
            cls = YoloUtils.softmax(predictor[1])
            rect = Rectangle((x, y), w, h, fill=False, color='#57d977',
                             linewidth=2)
            ax.add_patch(rect)

            labels = ['Square', 'Circle', 'Triangle', 'Unknown']
            ind = np.argmax(cls)
            label = '%s : (%.2f, %.2f, %.2f)' \
                    % (labels[ind], cls[0], cls[1], cls[2])
            t = plt.text(x - 5, y - 5, label, color='white', fontsize=12)
            t.set_bbox(dict(facecolor='#57d977', alpha=0.75,
                            edgecolor='#57d977'))

import cv2
import matplotlib.pyplot as plt
import numpy as np


class YoloValidation:
    def __init__(self):
        pass

    @staticmethod
    def _draw_bb_(bb1, bb2):
        """
        Plots the two bounding boxes on a blank image and displays the computed
        iou.

        Parameters
        ----------
        bb1 : tuple of floats
            A bounding box tuple of shape (x,y,w,h).
        bb2 : tuple of floats
            A bounding box tuple of shape (x,y,w,h).
        """
        # Compute bounding box of the union to figure out the dimensions of the
        # embedding frame.
        #
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        tl = (min(x1, x2), min(y1, y2))
        br = (max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2))
        w, h = (br[0] - tl[0], br[1] - tl[1])

        img = np.zeros((h + 1, w + 1, 3), dtype='uint8')
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 1)
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        ax = plt.gca()
        ax.set_title('Intersection: {0}, Union: {1}, IOU: {2}'
                     .format(YoloValidation.intersection(bb1, bb2),
                             YoloValidation.union(bb1, bb2),
                             YoloValidation.iou(bb1, bb2)))

    @staticmethod
    def union(bb1, bb2):
        """
        Computes the area of the union of two bounding boxes.

        Parameters
        ----------
        bb1 : tuple of floats
            A bounding box tuple of shape (x,y,w,h).
        bb2 : tuple of floats
            A bounding box tuple of shape (x,y,w,h).

        Returns
        -------
        float
            The area of the union of the two boxes.
        """
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        return (w1 * h1 + w2 * h2) - YoloValidation.intersection(bb1, bb2)

    @staticmethod
    def iou(bb1, bb2):
        """
        Computes the intersection over union of two bounding boxes.

        Parameters
        ----------
        bb1 : tuple of floats
             A bounding box tuple of shape (x,y,w,h).
        bb2 : tuple of floats
            A bounding box tuple of shape (x,y,w,h).
        Returns
        -------
        float
            The intersection over union of the two bounding boxes.
        """

        # The Union of both bounding boxes is: union(bb1,bb2) := area(bb1) +
        # area(bb2) - intersection(bb1,bb2)
        #
        i = YoloValidation.intersection(bb1, bb2)
        u = YoloValidation.union(bb1, bb2)

        return i / u

    @staticmethod
    def intersection(bb1, bb2):
        """
        Computes the area of the intersection of two bounding boxes.

        Parameters
        ----------
        bb1 : tuple of floats
             A bounding box tuple of shape (x,y,w,h).
        bb2 : tuple of floats
            A bounding box tuple of shape (x,y,w,h).

        Returns
        -------
        float
            The area of the intersection of the two bounding boxes.
        """

        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2

        # Compute width:
        #  - compute the right sides of both rectangles and pick the one on
        #  the left
        #  - compute the left sides of both rectangles and pick the one on
        #  the right
        #
        r1 = min(x1 + w1, x2 + w2)
        r2 = max(x1 + w1, x2 + w2)
        l1 = min(x1, x2)
        l2 = max(x1, x2)

        w = min(r1, r2) - max(l1, l2)
        if w < 0:
            w = 0

        # Compute height:
        #  - compute the top sides of both rectangles and pick the lower one
        #  - compute the bottom sides of both rectangles and pick the upper one
        #
        # Remember: y grows downwards
        #
        t1 = min(y1, y2)
        t2 = max(y1, y2)

        b1 = min(y1 + h1, y2 + h2)
        b2 = max(y1 + h1, y2 + h2)

        h = min(b1, b2) - max(t1, t2)

        if h < 0:
            h = 0
        return w * h

    @staticmethod
    def _best_iou(img_gt_bboxes, pred_box):
        """
        Computes the iou between a list of ground-truth boxes and a predicted
        box. Returns the best (highest) iou value and the corresponding best
        matching box within the ground-truth.

        Parameters
        ----------
        img_gt_bboxes : list of tuples
            A list of ground-truth bounding boxes
            i.e.: [((x,y,w,h), np.array([c0,...,cn])), ...].
        pred_box : tuple
            A single predicted bounding box as tuple
            i.e.: (conf, (x,y,w,h), np.array([c0,...,cn])).

        Returns
        -------
        tuple : (float, tuple)
            Returns a tuple (best_iou, best_matched_gt_box) where best_iou
            is the value of the highest iou, best_matched_gt_box is a tuple
            representing the best matched bounding box as
            ((x,y,w,h), np.array([c0,...,cn])).
        """

        best_iou = 0
        best_matched_gt_box = None

        for gt_box in img_gt_bboxes:

            if np.argmax(gt_box[1]) != np.argmax(pred_box[2]):
                continue

            iou = YoloValidation.iou(gt_box[0], pred_box[1])
            if iou > best_iou:
                best_iou = iou
                best_matched_gt_box = gt_box

        return best_iou, best_matched_gt_box

    @staticmethod
    def _is_duplicate(gt_bbox, pred_bbox, all_img_pred_bbox):
        """
        This method checks if the predicted bbox is a duplicate (i.e.:
        there is another box that predicts the same gt-box but with a higher
        iou). We need this to compute the precision/recall. If a box is a
        duplicate it is not regarded as a true positive.

        Parameters
        ----------
        gt_bbox : tuple
             Ground-truth bounding box i.e.: ((x,y,w,h), np.array([c0,..,cn])).
        pred_bbox : tuple
            Predicted box i.e: (conf, (x,y,w,h), np.array([c0,...,cn])).
        all_img_pred_bbox : list of tuples
            A list of all ground-truth bounding boxes for a single image,
            i.e.: [((x,y,w,h), np.array([c0,..,cn])), ...].

        Returns
        -------
        boolean
            True if pred_bbox is a duplicate for gt_bbox, False otherwise.
        """
        iou_list = list()
        for box in all_img_pred_bbox:
            iou = YoloValidation.iou(gt_bbox[0], box[1])
            iou_list.append((iou, box[1]))

        # Sort by iou
        iou_list = sorted(iou_list, key=lambda x: x[0], reverse=True)

        # Check if pred_bbox is the top box
        #
        top_box = iou_list[0][1]
        cur_box = pred_bbox[1]
        if abs(top_box[0] - cur_box[0]) < 1e-5 and \
                abs(top_box[1] - cur_box[1]) < 1e-5 and \
                abs(top_box[2] - cur_box[2]) < 1e-5 and \
                abs(top_box[3] - cur_box[3]) < 1e-5:
            return False

        return True

    @staticmethod
    def precision_recall(all_gt, all_pred, iou_t, class_list,
                         plot_results=True):
        """
        Computes and plots the precision-recall curve which gives us the
        precision value (y-axis) for a given recall value (x-axis).

        Parameters
        ----------
        all_gt : list of list of tuples
            The ground-truth bounding boxes for the entire batch of images. Each
            entry in the list corresponds to all bounding boxes of a single
            image. The ordering of all_gt and all_pred must match.
        all_pred : list of list of tuples
            The predicted bounding boxes for the entire batch of images. Each
            entry in the list corresponds to all bounding boxes of a single
            image. The ordering of all_gt and all_pred must match.
        iou_t : float
            Intersection over union threshold for deciding which prediction is
            a true positive.
        class_list : list of numpy.arrays
            Classes to take into account, these must be one-hot-encoded. The
            precision-recall curve is computed for each class separately and
            returned as a dictionary.
        plot_results : boolean, default=True
            If set to True, the precision-recall curves will be plotted.

        Returns
        -------
        dictionary: Keys - class-label, Values - (list of float, list of float)
            Stores the precision-recall curve for each class using the
            class-label (notice: not one-hot-encoded, it is an int use
            np.argmax() to convert) as key. The values for each key is a
            tuple of two lists (precision,recall).
        """

        result = dict()
        for classId in class_list:
            recall, precision = YoloValidation._precision_recall_single_class(
                all_gt, all_pred, iou_t, classId)
            # np.array is not hashable, we therefore store the class label
            #
            result[np.argmax(classId)] = (precision, recall)

        if plot_results:

            plt.figure(figsize=(6, 4))
            ax = plt.gca()
            plt.xlabel('Recall', fontsize=16)
            plt.ylabel('Precision', fontsize=16)

            for key in result.keys():
                precision = result[key][0]
                recall = result[key][1]
                plt.plot(recall, precision, label="Class: {0}".format(key))

            ax.legend()

        return result

    @staticmethod
    def _precision_recall_single_class(all_gt, all_pred, iou_t, class_id):
        """
        Computes the precision-recall curve for a single class.

        Parameters
        ----------
        all_gt : list of list of tuples
            The ground-truth bounding boxes for the entire batch of images. Each
            entry in the list corresponds to all bounding boxes of a single
            image. The ordering of all_gt and all_pred must match.
        all_pred : list of list of tuples
            The predicted bounding boxes for the entire batch of images. Each
            entry in the list corresponds to all bounding boxes of a single
            image. The ordering of all_gt and all_pred must match.
        iou_t : float
            Intersection over union threshold for deciding which prediction is
            a true positive.
        class_id : numpy.arrays - (n_classes)
            Class to take into account, this must be one-hot-encoded.

        Returns
        -------
        tuple of lists
            Returns a tuple (recall, precision) of two lists holding the
            recall and precision values.
        """

        ap_list = []
        num_objects = 0

        # We first have to compute the iou values for each box in the
        # predictions. This is needed to define whether it is correct or
        # not which is then used for computing the precision/recall curves
        # based on the confidence values.
        #
        for img_idx in range(len(all_pred)):
            for box_idx in range(len(all_pred[img_idx])):

                pred_box = all_pred[img_idx][box_idx]

                if np.argmax(pred_box[2]) != np.argmax(class_id):
                    continue

                iou, gt_box = YoloValidation._best_iou(all_gt[img_idx],
                                                       pred_box)
                is_tp = True

                if iou < iou_t:
                    is_tp = False
                elif YoloValidation._is_duplicate(gt_box, pred_box,
                                                  all_pred[img_idx]):
                    is_tp = False

                # Remember if it was correct and confidence, that is all we
                # need from here
                #
                ap_list.append((pred_box[0], is_tp))  # (confidence, correct)

        # remove wrong class predictions from flat list
        sorted_ap_list = sorted(ap_list, key=lambda x: x[0], reverse=True)

        # figure out the number of objects in the groundtruth
        for img_idx in range(len(all_gt)):
            for box_idx in range(len(all_gt[img_idx])):

                box = all_gt[img_idx][box_idx]

                if np.argmax(box[1]) == np.argmax(class_id):
                    num_objects = num_objects + 1

        # Compute precision and recalls from confidence sorted
        # correct/incorrect list
        # What we do here is we count the number of true from the top of the
        # list to the bottom of the list
        #
        # conf   correct         precision            recall
        #  1       true     #true / (rowindex+1)       # true / #(objects in
        #  dataset)
        # 0.99     false
        # 0.6      true
        #
        ranked = np.array(sorted_ap_list)
        precision = np.cumsum(ranked[:, 1]) / np.arange(1, ranked.shape[0] + 1)
        recall = np.cumsum(ranked[:, 1]) / num_objects

        return recall, precision

    @staticmethod
    def mean_average_precision(gt_bboxes, pred_bboxes, iou_threshold_list,
                               class_list):
        """
        Computes the mean average precision over all classes based on
        interpolated precision at 11 recall steps [0,0.1,...,0.9,1].

        Parameters
        ----------
        gt_bboxes : list of list of tuples
            Ground-truth bounding boxes for all images in the batch, i.e.:
            [ [((x,y,w,h), array([class_0,...,class_n])), ...], ...].
        pred_bboxes : list of list of tuples
            Predicted bounding boxes for all images in the batch, i.e.:
            [ [(conf, (x,y,w,h), array([class_0,..., class_n])), ...], ...].
            Ordering of gt_bboxes and pred_bboxes must match!
        iou_threshold_list : list of float
            List of iou thresholds used to compute the precision-recall curves,
            we then compute the mean over all the precision-recall curves over
            all classes to get the mean-average-precision.
        class_list : list of np.array
            Lists all classes to take into account for computing the map.
            Entries in the list must be one-hot-encoded, i.e.:
            [ [0,0,1], [1,0,0], ...].

        Returns
        -------
        float
            The mean average precision at all threshold in iou_threshold_list
            over all classes in class_list is returned.
        """

        ap_per_iou = list()
        ap_results = list()

        for iou_threshold in iou_threshold_list:

            results = YoloValidation.precision_recall(gt_bboxes, pred_bboxes,
                                                      iou_threshold, class_list,
                                                      False)
            ap_results = list()

            for class_id in class_list:

                precision = results[np.argmax(class_id)][0]
                recall = results[np.argmax(class_id)][1]

                recall_steps = np.arange(0, 1.1, 0.1)
                ap = 0
                for recall_interp in recall_steps:
                    # Find indices of recall_values close to the current
                    # interpolation step
                    #
                    indices = abs(recall - recall_interp) < 0.01
                    # If we do not have enough data, we can not simply pick a
                    # value close to
                    # the interpolation step, but we really have to interpolate.
                    #
                    if not np.any(indices):

                        # Search for the indices left and right to the
                        # recall_interp value.
                        #
                        recall_indices_left = (recall < recall_interp)
                        recall_indices_right = (recall > recall_interp)

                        # Check if we have a left index
                        #
                        if np.sum(recall_indices_left) == 0:
                            # Pick the last index on the left and the first
                            # on the right to get the two closest points
                            #
                            right_index = np.min(
                                np.argwhere(recall_indices_right))

                            # The interp value is smaller than any we have.
                            # We use sensitivity 1 at recall 0 and interpolate
                            # linearly.
                            p1 = 1.0  # precision 1
                            r1 = 0  # recall 0

                            p2 = precision[right_index]
                            r2 = recall[right_index]

                            print(
                                'WARN: No recall values on the left of {0}, '
                                'results might be inaccurate!'
                                .format(recall_interp))

                        else:
                            left_index = np.max(
                                np.argwhere(recall_indices_left))

                            if np.sum(recall_indices_right) == 0:

                                p2 = 0  # precision 1
                                r2 = 1  # recall 0

                                p1 = precision[left_index]
                                r1 = recall[left_index]

                                print(
                                    'WARN: No recall values on the right of '
                                    '{0}, results might be inaccurate!'
                                    .format(recall_interp))

                            else:

                                # Pick the last index on the left and the
                                # first on the right to get the two closest
                                # points
                                #
                                right_index = np.min(
                                    np.argwhere(recall_indices_right))

                                p1 = precision[left_index]
                                r1 = recall[left_index]

                                p2 = precision[right_index]
                                r2 = recall[right_index]

                        # interpolate linearly
                        #
                        interp_value = (recall_interp - r1) * (
                                    (p2 - p1) / (r2 - r1)) + p1
                        ap = ap + interp_value
                        print(
                            'WARN: interpolating {0} between recall={1} with '
                            'precision={2} -> {3}'
                            .format(recall_interp, (r1, r2), (p1, p2),
                                    interp_value))
                    else:
                        # Find maximum precision value
                        #
                        max_value = np.max(precision[indices])
                        ap = ap + max_value

                ap = ap / len(recall_steps)
                ap_results.append(ap)

        ap_per_iou.append(np.mean(ap_results))

        return np.mean(ap_per_iou)

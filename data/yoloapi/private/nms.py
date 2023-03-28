from yoloapi.public.validation import YoloValidation

__author__ = "Sebastian Hegenbart"


# NOT FOR STUDENTS EYES they have to implement this
#
def non_maximum_suppression_batch(bboxes_batch):
    """
    Todo: This is for the students to implement.
    Performs non maximum suppression on a list of list of bounding boxes as
    returned by YoloUtils.convert_yolotensor_to_bboxes().

    Parameters
    ----------
    bboxes_batch : list of list of tuples
        List of lists of bounding boxes (one for each image),
        i.e.: [ [(conf, (x,y,w,h), np.array([c0,...,cn]), ...], ...]
    Returns
    -------
        list of list of tuples
        The non-maximum-suppressed list of lists of bounding boxes.
    """
    batch_maximum_bboxes = list()

    for idx in range(len(bboxes_batch)):

        bboxes = bboxes_batch[idx]
        # Sort boxes by confidence score.
        #
        maximum_bboxes = list()
        bboxes_sorted = sorted(bboxes, key=lambda i: i[0], reverse=True)

        # Pick first it will be our first object
        #
        while len(bboxes_sorted) > 0:

            cur_maximum = bboxes_sorted[0]

            # Remove the first box
            #
            bboxes_sorted.remove(cur_maximum)
            maximum_bboxes.append(cur_maximum)

            # Remove boxes with overlap > 50%, do not remove things from
            # collections while iterating them.
            #
            remove_list = list()
            for bbox in bboxes_sorted:

                iou = YoloValidation.iou(cur_maximum[1], bbox[1])
                if iou > 0.5:
                    remove_list.append(bbox)

            for bbox in remove_list:
                bboxes_sorted.remove(bbox)

        batch_maximum_bboxes.append(maximum_bboxes)
    return batch_maximum_bboxes

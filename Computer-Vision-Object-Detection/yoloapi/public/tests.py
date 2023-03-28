import tensorflow as tf
import pickle


class YoloSimpleLossTester:
    """
        Test class for the simple yolo loss.
    """

    def __init__(self):
        self.test_data_file = \
            'yoloapi/data/test-data/yolo_simple_loss_testdata.pickle'
        self.test_data = self._load_test_data__()

    def _load_test_data__(self):
        with open(self.test_data_file, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp

    def test_class_loss(self, loss_func, is_tensorflow=False):
        """
        Tests the class loss function.

        Parameters
        ----------
        loss_func - func
            A reference to the loss_function to test against.
        is_tensorflow - boolean
            True indicates the tested function operates on tf.Tensor,
            False indicates the tested function operates on numpy.array.

        Returns
        -------
        boolean
            True on all tests passed, False otherwise.

        """
        return self.__run_test__(loss_func, 'class_loss', is_tensorflow)

    def test_confidence_loss(self, loss_func, is_tensorflow=False):
        """
        Tests the class loss function.

        Parameters
        ----------
        loss_func - func
           A reference to the loss_function to test against.
        is_tensorflow - boolean
           True indicates the tested function operates on tf.Tensor,
           False indicates the tested function operates on numpy.array.

        Returns
        -------
        boolean
           True on all tests passed, False otherwise.

        """
        return self.__run_test__(loss_func, 'confidence_loss', is_tensorflow)

    def test_location_loss(self, loss_func, is_tensorflow=False):
        """
        Tests the class loss function.

        Parameters
        ----------
        loss_func - func
           A reference to the loss_function to test against.
        is_tensorflow - boolean
           True indicates the tested function operates on tf.Tensor,
           False indicates the tested function operates on numpy.array.

        Returns
        -------
        boolean
           True on all tests passed, False otherwise.

        """
        return self.__run_test__(loss_func, 'location_loss', is_tensorflow)

    def test_shape_loss(self, loss_func, is_tensorflow=False):
        """
        Tests the class loss function.

        Parameters
        ----------
        loss_func - func
           A reference to the loss_function to test against.
        is_tensorflow - boolean
           True indicates the tested function operates on tf.Tensor,
           False indicates the tested function operates on numpy.array.

        Returns
        -------
        boolean
           True on all tests passed, False otherwise.

        """
        return self.__run_test__(loss_func, 'shape_loss', is_tensorflow)

    def test_yolo_loss(self, loss_func, is_tensorflow=False):
        """
        Tests the class loss function.

        Parameters
        ----------
        loss_func - func
           A reference to the loss_function to test against.
        is_tensorflow - boolean
           True indicates the tested function operates on tf.Tensor,
           False indicates the tested function operates on numpy.array.

        Returns
        -------
        boolean
           True on all tests passed, False otherwise.

        """
        return self.__run_test__(loss_func, 'combined_loss', is_tensorflow)

    def __run_test__(self, loss_func, key, is_tensorflow):
        success = True
        for i in range(len(self.test_data['combined_loss'])):

            z = self.test_data['Z'][i]
            zhat = self.test_data['Zhat'][i]

            if is_tensorflow:
                z = tf.convert_to_tensor(z)
                zhat = tf.convert_to_tensor(zhat)

            loss_test = loss_func(z, zhat)

            if is_tensorflow:
                loss_test = loss_test.numpy()

            loss_ref = self.test_data[key][i]

            if abs(loss_test - loss_ref) > 1e-3:
                print(
                    'FAIL: Index {0}: expected={1}, got={2}'.format(i, loss_ref,
                                                                    loss_test))
                success = False
        if success:
            print('SUCCESS: All tests passed!')
        else:
            print('FAIL: Some tests did not pass!')

        return success


class IOUTester:
    """
    This class is used to test the iou implementation.
    """

    def __init__(self):
        self.test_data_file = 'yoloapi/data/test-data/iou_testdata.pickle'
        self.test_data = self._load_test_data()

    def _load_test_data(self):
        with open(self.test_data_file, 'rb') as handle:
            self.test_data = pickle.load(handle)
            return self.test_data

    def test_iou(self, iou_func):
        """
        Tests the iou function.

        Parameters
        ----------
        iou_func - func
            A reference to the iou function.

        Returns
        -------
        boolean
            True if all tests pass, False otherwise.
        """
        success = True

        for idx in range(len(self.test_data['iou'])):
            iou_real = iou_func(self.test_data['bbox1'][idx],
                                self.test_data['bbox2'][idx])
            if abs(iou_real - self.test_data['iou'][idx]) > 1e-5:
                print('FAIL: bb1 = {0}, bb2 = {1}, expected iou = {2}, '
                      'actual iou = {3}'
                      .format(self.test_data['bbox1'][idx],
                              self.test_data['bbox2'][idx],
                              self.test_data['iou'][idx], iou_real))
                success = False

        if success:
            print('SUCCESS: All tests passed!')

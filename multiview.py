import numpy as np
import cv2 as cv
import nibabel as nib
from scipy.ndimage import zoom


def mouse_handler(event, x, y, flags, param):
    '''
    OpenCV mouse event handler
    '''
    current_point, volume, window_name = param["point"], param["volume"], param["window_name"]
    x = x//param["display_resize"]
    y = y//param["display_resize"]

    if ((event == cv.EVENT_LBUTTONDOWN or param["dragging"]) and param["previous_x"] != x
            and param["previous_y"] != y) or event == cv.EVENT_MBUTTONDOWN:

        window = param["get_current_window"][x]
        param["dragging"] = True
        param["previous_x"] = x
        param["previous_y"] = y

        current_point[window] += (flags == 4)*1 - (flags == 12)*1

        if window == 0:
            current_point = [current_point[0], y, x]
        elif window == 1:
            current_point = [y, current_point[1], x - param["cube_size"]]
        elif window == 2:
            current_point = [y, x - param["cube_size"]*2, current_point[2]]
        param["point"] = current_point

        if x is not None and y is not None:
            axis0 = np.copy(volume[current_point[0], :, :])
            axis1 = np.copy(volume[:, current_point[1], :])
            axis2 = np.copy(volume[:, :, current_point[2]])

            axis0 = cv.circle(axis0, (current_point[2], current_point[1]), 2, 1)
            axis1 = cv.circle(axis1, (current_point[2], current_point[0]), 2, 1)
            axis2 = cv.circle(axis2, (current_point[1], current_point[0]), 2, 1)

            axis0 = cv.line(axis0, (0, current_point[1]), (param["cube_size"] - 1, current_point[1]), 1)
            axis0 = cv.line(axis0, (current_point[2], 0), (current_point[2], param["cube_size"] - 1), 1)

            axis1 = cv.line(axis1, (current_point[2], 0), (current_point[2], param["cube_size"] - 1), 1)
            axis1 = cv.line(axis1, (0, current_point[0]), (param["cube_size"] - 1, current_point[0]), 1)

            axis2 = cv.line(axis2, (current_point[1], 0), (current_point[1], param["cube_size"] - 1), 1)
            axis2 = cv.line(axis2, (0, current_point[0]), (param["cube_size"] - 1, current_point[0]), 1)

        display = np.hstack((axis0, axis1, axis2))
        cv.imshow(window_name, cv.resize(display, (0, 0), fx=param["display_resize"], fy=param["display_resize"]))
    elif event == cv.EVENT_LBUTTONUP:
        param["dragging"] = False


class MultiViewer():
    def __init__(self, volume, mask=None, normalize=False, window_name="MultiViewer", cube_side=200, resize_factor=2):
        if normalize:
            self.volume = (volume - volume.min()) / (volume.max() - volume.min())
        else:
            self.volume = volume

        self.volume_shape = volume.shape

        if self.volume_shape != (cube_side, cube_side, cube_side):
            zoom_factors = (cube_side/self.volume_shape[0], cube_side/self.volume_shape[1], cube_side/self.volume_shape[2])
            self.volume = zoom(self.volume, zoom_factors, order=5)
            if mask is not None:
                self.volume += zoom(mask, zoom_factors, order=0)
                self.volume[self.volume > 1] = 1
            self.volume_shape = self.volume.shape
            assert self.volume_shape == (cube_side, cube_side, cube_side)

        self.current_point = (np.array(self.volume_shape)/2).astype(np.int)
        self.window_name = window_name
        self.resize_factor = resize_factor
        cv.namedWindow(self.window_name)
        get_current_window = np.concatenate((np.zeros(self.volume_shape[0]),
                                             np.ones(self.volume_shape[1]),
                                             2*np.ones(self.volume_shape[2]))).astype(np.uint8)
        handler_param = {"get_current_window": get_current_window, "window_name": self.window_name, "volume": self.volume,
                         "point": self.current_point, "cube_size": cube_side, "previous_x": cube_side/2,
                         "previous_y": cube_side/2, "dragging": False, "display_resize": resize_factor}
        cv.setMouseCallback(window_name, mouse_handler, param=handler_param)

    def display(self):
        axis0 = self.volume[self.current_point[0], :, :]
        axis1 = self.volume[:, self.current_point[1], :]
        axis2 = self.volume[:, :, self.current_point[2]]
        cv.imshow(self.window_name, cv.resize(np.hstack((axis0, axis1, axis2)), (0, 0),
                                              fx=self.resize_factor, fy=self.resize_factor))
        key = cv.waitKey(0)
        if key == 27:
            return 'ESC'
        else:
            cv.destroyWindow(self.window_name)


if __name__ == "__main__":
    from sys import argv
    import os
    try:
        print("Loading...")
        path = argv[1]
        fname = os.path.basename(path)

        if fname.split('.')[1] == 'npz':
            test_vol = np.load(path)
            MultiViewer(test_vol["img"], mask=test_vol["tgt"]).display()
        else:
            test_vol = nib.load(argv[1]).get_fdata()
            MultiViewer(test_vol, normalize=True).display()
    except IndexError:
        print("Give a path to a file to multiview (NifT, MNC)")

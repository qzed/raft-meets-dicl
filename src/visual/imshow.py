import cv2

from . import flow_mb as flowvis
from . import flow_dark as flowvis_dark


class ImageWindow:
    def __init__(self, title):
        self.title = title

    def wait(self):
        # Using OpenCV imshow with waitKey(0) can deadlock if the window is
        # closed by pressing the 'x' button: It still waits for a key, but it
        # can't get any input anymore due to the window being gone. This
        # wouldn't be too annoying if OpenCV didn't also block any attemtps to
        # Ctrl-C waitKey(0). Work around this by repeatedly checking if the
        # window is still there. This makes it behave nicely to both closing
        # the window and Ctrl-C.

        while cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(250) != -1:
                break


def show_image(title, rgb):
    cv2.imshow(title, rgb[:, :, ::-1])
    return ImageWindow(title)


def show_flow(title, flow, *args, **kwargs):
    flow = flowvis.flow_to_rgba(flow, *args, **kwargs)

    return show_image(title, flow)


def show_flow_dark(title, flow, *args, **kwargs):
    flow = flowvis_dark.flow_to_rgba(flow, *args, **kwargs)

    return show_image(title, flow)

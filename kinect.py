import cv2
import numpy as np

import pyk4a

from pyk4a import Config, PyK4A

# in mm, specify second operand in feet
MIN_DIST = 300 * 1.5
MAX_DIST = 300 * 7
COLORMAP = cv2.COLORMAP_OCEAN


def colorize(
    image: np.ndarray,
    clipping_range=(None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

import os

class KinectCam:
    def __init__(self, capture=False, filename=""):
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.OFF,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=False,
            )
        )
        k4a.start()
        k4a.whitebalance = 4510
        self.cam = k4a

        self.optic_flow = cv2.DISOpticalFlow.create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
        )
        self.optic_flow.setUseSpatialPropagation(True)
        # self.prev_flow = None
        self.prev_frame = None
        self.frame_count = 0

        self.capture = capture
        self.filename = filename

        if capture:
            os.makedirs(f"images/{filename}", exist_ok=True)

    def get_frame(self):
        capture = self.cam.get_capture()

        if capture.depth is None:
            return None, None, None

        translation_matrix = np.array(
            [[1, 0, (1280 - 800) / 2], [0, 1, 0]], dtype=np.float32
        )

        scale_factor = 720 / capture.depth.shape[1]
        depth_img = cv2.resize(capture.depth, (800, 720), scale_factor, scale_factor)
        depth_img = cv2.warpAffine(
            src=depth_img, M=translation_matrix, dsize=(1280, 720)
        )

        

        # Based in mm of depth camera
        mask = cv2.inRange(depth_img, MIN_DIST, MAX_DIST)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

        depth_flow_img = colorize(depth_img, (MIN_DIST, MAX_DIST), COLORMAP)


        if self.capture:
            cv2.imwrite(f"images/{self.filename}/frame_{self.frame_count:04d}.png", depth_flow_img)
            self.frame_count += 1

        if self.prev_frame is None:
            self.prev_frame = depth_flow_img

        # if self.prev_flow is not None:
        #     flow = self.optic_flow.calc(
        #         self.prev_frame,
        #         depth_flow_img,
        #         warp_flow(self.prev_flow, self.prev_flow),
        #     )
        # else:
        #     flow = self.optic_flow.calc(self.prev_frame, depth_flow_img, None)

        # props = self.get_flow_props(flow)

        # self.prev_flow = flow
        self.prev_frame = depth_flow_img

        # masked_img = cv2.bitwise_and(color_img, color_img, mask=mask)

        # n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        #     mask, 12, cv2.CV_32S
        # )

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        max_contour = None

        if len(contours) != 0:
            max_area = cv2.contourArea(contours[0])
            max_contour = contours[0]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # Extrapolate center of contour
            m = cv2.moments(max_contour)

            try:
                centroids = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))]
            except ZeroDivisionError:
                return None, None, None

            if max_area < 20000:
                centroids = []
                max_contour = None

        mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if max_contour is not None:
            max_contour = np.vstack(max_contour).squeeze()

        return depth_flow_img, centroids, max_contour

    def close(self):
        self.cam.stop()

    # def get_flow_props(self, flow):
    #     u, v = cv2.split(flow)
    #     mu = cv2.mean(u)
    #     mv = cv2.mean(v)

    #     height = self.prev_frame.shape[0]
    #     width = self.prev_frame.shape[1]
    #     X = np.fromfunction(lambda y, x: x, (height, width))
    #     Y = np.fromfunction(lambda y, x: y, (height, width))

    #     mag, angle = cv2.cartToPolar(u, v)
    #     mx = np.sum(mag * X) / np.sum(mag)
    #     my = np.sum(mag * Y) / np.sum(mag)

    #     return dict(
    #         mean_x=mu[0] / self.prev_frame.shape[1],
    #         mean_y=mv[0] / self.prev_frame.shape[0],
    #         motion_point_x=mx / self.prev_frame.shape[1],
    #         motion_point_y=my / self.prev_frame.shape[0],
    #     )


if __name__ == "__main__":
    main()

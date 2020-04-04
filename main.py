import cv2 as cv
import numpy as np
# from PIL import Image
import os


class PaperStepSequencer:
    def __init__(self):
        # load predefined dictionary
        self.ar_ids = [11, 22, 33, 44]
        self.aruco_dict = PaperStepSequencer.init_markers(self.ar_ids)
        # Initialize the detector parameters using default values
        self.aruco_param = cv.aruco.DetectorParameters_create()

        self.world_h = 480
        self.world_w = 710
        self.corners_world = np.array([
            [0, 0],
            [self.world_w, 0],
            [self.world_w, self.world_h],
            [0, self.world_h]
        ])


    @staticmethod
    def init_markers(ar_ids):
        # initialise directories
        markers_dir = "output/markers"
        if not os.path.exists(markers_dir):
            os.makedirs(markers_dir)

        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        for ar_id in ar_ids:
            # Generate the marker
            markerImage = np.zeros((200, 200), dtype=np.uint8)
            cv.aruco.drawMarker(dictionary, ar_id, 200, markerImage, 1)
            cv.imwrite(f"{markers_dir}/marker{ar_id}.png", markerImage)
        return dictionary

    @staticmethod
    def detect_screen_corners(frame, aruco_dict, aruco_param, ar_ids):
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_param)

        if markerIds is None or len(markerIds) == 0:
            return None

        # filter out unknown markers
        inds = [i for i, mid in enumerate(markerIds) if mid[0] in ar_ids]
        markerCorners = [markerCorners[ind] for ind in inds]
        markerIds = [markerIds[ind] for ind in inds]
        if len(markerIds) != 4:
            return None
        if any([ar_id not in markerIds for ar_id in ar_ids]):
            return None

        # markerCorners, markerIds = detect_markers(frame, aruco_dict, ar_ids)
        markerCorners = [c[0] for c in markerCorners]
        markerIds = [i[0] for i in markerIds]
        corners_screen = []
        for i, ar_id in enumerate(ar_ids):
            ind = markerIds.index(ar_id)
            corners_screen.append(markerCorners[ind][i])
        corners_screen = np.array(corners_screen)
        return corners_screen

    @staticmethod
    def draw_screen_feedback(frame, corners_screen, corners_world, world_w, world_h):
        frame = frame.copy()

        s_from_w, status = cv.findHomography(corners_world, corners_screen)
        lines = []
        for x in range(0, world_w + 1, 10):
            pts_world = np.array([[x, 0, 1], [x, world_h, 1]])

            pts_screen = np.dot(s_from_w, pts_world.transpose()).transpose()
            pts_screen = pts_screen / pts_screen[:, 2:3]
            pts_screen = pts_screen[:, :2]

            line = np.array(pts_screen, dtype=np.int32)
            lines.append(line)
        for y in range(0, world_h + 1, 10):
            pts_world = np.array([[0, y, 1], [world_w, y, 1]])

            pts_screen = np.dot(s_from_w, pts_world.transpose()).transpose()
            pts_screen = pts_screen / pts_screen[:, 2:3]
            pts_screen = pts_screen[:, :2]

            line = np.array(pts_screen, dtype=np.int32)
            lines.append(line)

        frame = np.stack([frame, frame, frame], axis=2)
        cv.polylines(frame, lines, True, (0, 255, 0))

        # cv.imshow("screen_feedback.png", frame)
        # cv.waitKey(0)
        # cv.imwrite("output/screen_feedback.png", frame)
        return frame

    def run_offline(self):
        # get src image and store point source coordinate system
        # frame = np.asarray(Image.open("frames/frame0.jpg").convert('L'))
        frame = cv.imread("frames/frame0.jpg", 0)

        self.process_frame(frame)

    def process_frame(self, frame):
        # detect area corner on the screen
        corners_screen = PaperStepSequencer.detect_screen_corners(
            frame, self.aruco_dict, self.aruco_param, self.ar_ids)
        if corners_screen is None:
            return frame

        # draw grid on frame to make sure the area is well detected
        frame_feedback = PaperStepSequencer.draw_screen_feedback(
            frame, corners_screen, self.corners_world, self.world_w, self.world_h)

        # world to screen homography
        w_from_s, status = cv.findHomography(corners_screen, self.corners_world)
        # Warp source image to destination based on homography
        frame_warped = cv.warpPerspective(frame, w_from_s, (self.world_w, self.world_h))

        # cv.imshow("warped.png", frame_warped)
        # cv.waitKey(0)
        # cv.imwrite("output/warped.png", frame_warped)

        return frame_feedback

    def run(self):
        cam = cv.VideoCapture(0)

        cv.namedWindow("test")

        img_counter = 0

        while True:
            ret, frame = cam.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_feedback = self.process_frame(frame)

            cv.imshow("test", frame_feedback)
            if not ret:
                break
            k = cv.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = f"opencv_frame_{img_counter}.png"
                cv.imwrite(img_name, frame)
                print(f"{img_name} written!")
                img_counter += 1

        cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    pss = PaperStepSequencer()
    pss.run()
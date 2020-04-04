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

        if markerIds is None:
            print("markerIds is None")
            return None
        elif len(markerIds)==0:
            print("len(markerIds)==0")
            return None

        # filter out unknown markers
        inds = [i for i, mid in enumerate(markerIds) if mid[0] in ar_ids]
        markerCorners = [markerCorners[ind] for ind in inds]
        markerIds = [markerIds[ind] for ind in inds]

        markerCorners = [c[0] for c in markerCorners]
        markerIds = [i[0] for i in markerIds]

        if len(markerIds) != 4:
            print("missing:", set(ar_ids).difference(markerIds))
            return None
        if any([ar_id not in markerIds for ar_id in ar_ids]):
            print("any([ar_id not in markerIds for ar_id in ar_ids])")
            return None

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
        cv.namedWindow("Camera")
        cv.namedWindow("Warped")

        # get src image and store point source coordinate system
        # frame = np.asarray(Image.open("frames/frame0.jpg").convert('L'))
        frame = cv.imread("frames/frame0.jpg", 0)
        frame_feedback, frame_warped = self.process_frame(frame)

        cv.imshow("Camera", frame_feedback)
        cv.imshow("Warped", frame_warped)
        cv.waitKey(0)

    def detect_coins(self, warped):
        warped = warped.copy()
        # Blur the image to reduce noise
        img_blur = cv.medianBlur(warped, 1)
        # Apply hough transform on the image
        circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img_blur.shape[0] / 64, param1=200, param2=10,
                                   minRadius=20, maxRadius=25)
        warped = img_blur
        warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))[0]
            centers =

            for i in circles[0, :]:
                # Draw outer circle
                cv.circle(warped, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                cv.circle(warped, (i[0], i[1]), 2, (0, 0, 255), 3)
        return warped

    def process_frame(self, frame):
        # detect area corner on the screen
        corners_screen = PaperStepSequencer.detect_screen_corners(
            frame, self.aruco_dict, self.aruco_param, self.ar_ids)
        if corners_screen is None:
            return None

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

        return frame_feedback, frame_warped

    def run(self):
        cam = cv.VideoCapture(0)

        cv.namedWindow("Camera")
        cv.namedWindow("Warped")

        img_counter = 0

        update_requested = False
        frame_is_valid = True

        ret, frame = cam.read()

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            ret = self.process_frame(frame)
            frame_is_valid = ret is not None
            if frame_is_valid:
                frame_feedback, frame_warped = ret
                cv.imshow("Camera", frame_feedback)
                cv.imshow("Warped", frame_warped)
            else:
                cv.imshow("Camera", frame)

            if update_requested and frame_is_valid:
                update_requested = False
                img_name = f"opencv_frame_{img_counter}.png"
                cv.imwrite(img_name, frame)
                print(f"{img_name} written!")
                img_counter += 1

            k = cv.waitKey(50)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                update_requested = True


        cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    pss = PaperStepSequencer()
    pss.run()
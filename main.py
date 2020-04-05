import cv2 as cv
import numpy as np
# from PIL import Image
import os
from threading import Thread
import time


class PaperStepSequencer:
    def __init__(self):
        # load predefined dictionary
        self.ar_ids = [11, 22, 33, 44]
        # self.aruco_dict = PaperStepSequencer.init_markers(self.ar_ids)
        PaperStepSequencer.init_markers(self.ar_ids)
        # Initialize the detector parameters using default values
        # self.aruco_param = cv.aruco.DetectorParameters_create()

        self.world_h = 480
        self.world_w = 710
        self.margin = 60
        self.corners_world = np.array([
            [0, 0],
            [self.world_w, 0],
            [self.world_w, self.world_h],
            [0, self.world_h]
        ])

        # top left corner of the grid
        self.grid_pos_xy = np.array([160, 180])
        # size in warped pixel of a grid square
        self.grid_square_size = 60
        # dimensionality of the grid
        self.grid_dim_xy = np.array([8, 4])
        self.grid = np.zeros(self.grid_dim_xy.transpose(), dtype=np.uint8)

        self.bpm = 120
        self.steps_per_beats = 4
        self.sequencer_prev_step = -1


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
        # return dictionary

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
            # print("missing:", set(ar_ids).difference(markerIds))
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

    @staticmethod
    def detect_coins(warped, margin):
        warped = warped.copy()
        h, w = warped.shape[0], warped.shape[1]

        # Blur the image to reduce noise
        img_blur = cv.medianBlur(warped, 5)
        # Apply hough transform on the image
        circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img_blur.shape[0] / 64, param1=200, param2=10,
                                   minRadius=20, maxRadius=30) # TODO: radius function of cm
        warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
        centers = []
        if circles is None:
            return warped, centers
        circles = circles[0]

        # filter out circle of which center is out of the ROI
        radius = []
        centers = []
        for c in circles:
            cx, cy, r = c
            if not margin < cx < w - margin:
                continue
            if not margin < cy < h - margin:
                continue
            centers.append([cx, cy])
            radius.append(r)

        # join overlapping circle (when centers overlapp another circle)
        restart = True
        while restart:
            restart = False
            for i in range(len(radius)-1):
                for j in range(i+1, len(radius)):
                    distance = ((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)**0.5
                    if distance < max(radius[i], radius[j]):
                        r = max(radius[i], radius[j])
                        cx = 0.5*centers[i][0] + 0.5*centers[j][0]
                        cy = 0.5*centers[i][1] + 0.5*centers[j][1]
                        radius = [radius[k] for k in range(len(radius)) if k != i and k != j]
                        centers = [centers[k] for k in range(len(centers)) if k != i and k != j]
                        radius.append(r)
                        centers.append([cx, cy])
                        restart = True
                        break
                if restart:
                    break

        # Draw detected circles
        radius = np.uint16(np.around(radius))
        centers = np.uint16(np.around(centers))
        for i in range(len(radius)):
            r = radius[i]
            cx, cy = centers[i]
            # Draw outer circle
            cv.circle(warped, (cx, cy), r, (0, 255, 0), 2)
            # Draw inner circle
            cv.circle(warped, (cx, cy), 2, (0, 0, 255), 3)
        return warped, centers

    def get_grid_inputs(self, centers, frame_warped):
        frame_tmp = frame_warped.copy()

        entries = np.array(centers)
        entries = (entries - self.grid_pos_xy) / self.grid_square_size
        entries = np.int32(np.floor(entries))

        entries += 1
        entries = [en for en in entries if min(en) >= 1]
        entries = [en for en in entries if en[0] <= self.grid_dim_xy[0]]
        entries = [en for en in entries if en[1] <= self.grid_dim_xy[1]]

        # recover top left corner of highlighted squares
        squares_pos = np.array(entries) -1
        squares_pos = squares_pos * self.grid_square_size + self.grid_pos_xy
        # compute coordinates of 4 corners
        squares = []
        for sq in squares_pos:
            x, y = np.int32(sq)
            d = np.int32(self.grid_square_size)
            square_corners = [[  x,   y],
                       [x+d,   y],
                       [x+d, y+d],
                       [  x, y+d]
                    ]
            squares.append(np.array(square_corners))
        cv.polylines(frame_tmp, squares, True, (255, 0, 0))
        frame_warped = np.uint8(0.5*np.float32(frame_warped) + 0.5*np.float32(frame_tmp))

        entries = [list(en) for en in entries]
        return entries, frame_warped

    def draw_current_step(self, frame_warped):
        frame_tmp = frame_warped.copy()

        # lets highlight a column:
        # recover top left corner of column
        x = self.grid_pos_xy[0] + self.sequencer_prev_step * self.grid_square_size
        print("Camera self.sequencer_prev_step", self.sequencer_prev_step)
        y = self.grid_pos_xy[1]
        dx = self.grid_square_size
        dy = self.grid_square_size * self.grid_dim_xy[1]
        rectangle_corners = np.array([
            [     x,      y],
            [x + dx,      y],
            [x + dx, y + dy],
            [     x, y + dy]
        ])

        cv.polylines(frame_tmp, [rectangle_corners], True, (0, 0, 255))
        frame_warped = np.uint8(0.5 * np.float32(frame_warped) + 0.5 * np.float32(frame_tmp))
        return frame_warped

    def run_midi(self):
        grid_length = self.grid_dim_xy[0]

        # unit in seconds
        beat_period = 60 / self.bpm  # seconds (0.5)
        one_step_period = beat_period / self.steps_per_beats  # (0.125)
        full_grid_period = one_step_period * grid_length  # (2.0)

        tss = []
        while True:
            ts = time.time()
            step_ts = (ts % full_grid_period) / one_step_period
            step = int(step_ts)
            if step == self.sequencer_prev_step:
                step += 1
            if step % grid_length != (self.sequencer_prev_step + 1) % grid_length:
                print("ERROR")
            self.sequencer_prev_step = step
            to_wait = (step + 1 - step_ts) * one_step_period

            # cv.waitKey(int(to_wait * 1000))
            time.sleep(to_wait)

            ts2 = time.time()
            tss.append(ts2)
            print("Sequencer self.sequencer_prev_step", self.sequencer_prev_step)
            # print("step:", self.sequencer_prev_step, ", to wait: ", to_wait)

    def process_frame(self, frame):
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        aruco_param = cv.aruco.DetectorParameters_create()

        # detect area corner on the screen
        corners_screen = PaperStepSequencer.detect_screen_corners(
            frame, aruco_dict, aruco_param, self.ar_ids)
        if corners_screen is None:
            return None

        # draw grid on frame to make sure the area is well detected
        frame_feedback = PaperStepSequencer.draw_screen_feedback(
            frame, corners_screen, self.corners_world, self.world_w, self.world_h)

        # world to screen homography
        w_from_s, status = cv.findHomography(corners_screen, self.corners_world)
        # Warp source image to destination based on homography
        margin = 60
        frame_warped = cv.warpPerspective(frame, w_from_s, (self.world_w, self.world_h))

        frame_warped, centers = PaperStepSequencer.detect_coins(frame_warped, self.margin)
        if len(centers) > 0:
            self.entries, frame_warped = self.get_grid_inputs(centers, frame_warped)

        # TODO: update squencer inputs on most recent available image
        frame_warped = self.draw_current_step(frame_warped)

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

            k = cv.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                update_requested = True


        cam.release()
        cv.destroyAllWindows()


def main():
    pss = PaperStepSequencer()
    # pss.run_offline()

    midi_process = Thread(target=pss.run_midi)
    midi_process.start()

    cv_process = Thread(target=pss.run)
    cv_process.start()
    midi_process.join()
    cv_process.join()

if __name__ == '__main__':
    main()

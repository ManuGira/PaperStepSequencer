import cv2 as cv
import numpy as np
# from PIL import Image
import os
from threading import Thread
import steprunner


class PaperStepSequencer:
    def __init__(self):
        # load predefined dictionary
        self.ar_ids = [11, 22, 33, 44]
        # self.aruco_dict = PaperStepSequencer.init_markers(self.ar_ids)
        PaperStepSequencer.init_markers(self.ar_ids)
        # Initialize the detector parameters using default values
        # self.aruco_param = cv.aruco.DetectorParameters_create()

        self.w_from_s = np.diag([1.0, 1.0, 1.0])
        self.s_from_w = np.diag([1.0, 1.0, 1.0])

        self.pixpermm = 8
        self.world_h = 48*self.pixpermm
        self.world_w = 71*self.pixpermm
        self.marker_size = 6*self.pixpermm
        # the order of the markers must match their ids
        self.markers_world_posxy = np.array([
            [                            0,                             0],
            [self.world_w-self.marker_size,                             0],
            [self.world_w-self.marker_size, self.world_h-self.marker_size],
            [                            0, self.world_h-self.marker_size]
        ])
        self.markers_corners_world = []
        for posxy in self.markers_world_posxy:
            x, y = posxy
            d = self.marker_size
            corners = [[    x,     y],
                       [x + d,     y],
                       [x + d, y + d],
                       [    x, y + d]]
            self.markers_corners_world.append(corners)

        self.stepRunner = steprunner.StepRunner(self.pixpermm, (self.world_w, self.world_h))

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

    def detect_screen_corners(self, frame, aruco_dict, aruco_param, ar_ids):
        # Detect the markers in the image
        marker_corners, detected_ids, rejectedCandidates = cv.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_param)

        if detected_ids is None:
            print("markerIds is None")
            return None, None
        elif len(detected_ids)==0:
            print("len(markerIds)==0")
            return None, None

        # filter out unkown markers
        inds = [i for i, mid in enumerate(detected_ids) if mid[0] in ar_ids]
        marker_corners = [marker_corners[ind] for ind in inds]
        detected_ids = [detected_ids[ind] for ind in inds]

        if len(detected_ids)<=1:
            print("after filter len(markerIds)==0")
            return None, None

        # convert to flatten list
        marker_corners = [c[0].tolist() for c in marker_corners]
        detected_ids = [i[0] for i in detected_ids]

        if any([ar_ids.count(ar_id) > 1 for ar_id in ar_ids]):
            print("An aruco marker has been detected more than once")
            return None, None

        markers_corners_screen = []
        markers_corners_world = []
        for i, detected_id in enumerate(detected_ids):
            ind = ar_ids.index(detected_id)
            markers_corners_screen += marker_corners[i]
            markers_corners_world += self.markers_corners_world[ind]
        markers_corners_screen = np.array(markers_corners_screen)
        markers_corners_world = np.array(markers_corners_world)
        return markers_corners_screen, markers_corners_world

    def draw_screen_feedback(self, frame, corners_screen, corners_world, world_w, world_h):
        frame = frame.copy()

        s_from_w, status = cv.findHomography(corners_world, corners_screen)
        self.s_from_w = self.s_from_w*0.9 + 0.1*s_from_w
        lines = []
        for x in range(0, world_w + 1, self.pixpermm):
            pts_world = np.array([[x, 0, 1], [x, world_h, 1]])

            pts_screen = np.dot(self.s_from_w, pts_world.transpose()).transpose()
            pts_screen = pts_screen / pts_screen[:, 2:3]
            pts_screen = pts_screen[:, :2]

            line = np.array(pts_screen, dtype=np.int32)
            lines.append(line)
        for y in range(0, world_h + 1, self.pixpermm):
            pts_world = np.array([[0, y, 1], [world_w, y, 1]])

            pts_screen = np.dot(self.s_from_w, pts_world.transpose()).transpose()
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

        # filter out centers out of grid
        minx, miny = self.stepRunner.grid_pos_xy
        maxx, maxy = self.stepRunner.grid_pos_xy+self.stepRunner.grid_size_xy
        centers = [c for c in centers if minx < c[0] < maxx and miny < c[1] < maxy]
        if len(centers) == 0:
            return [], frame_warped

        centers = np.array(centers)
        entry_rows = (centers[:, 1] - self.stepRunner.grid_pos_xy[1]) / self.stepRunner.grid_square_size_y
        entry_rows = np.int32(np.floor(entry_rows))

        entry_steps = (centers[:, 0] - self.stepRunner.grid_pos_xy[0]) / self.stepRunner.grid_square_size_x[entry_rows]
        entry_steps = np.int32(np.floor(entry_steps))

        entry_rows.shape += (1,)
        entry_steps.shape += (1,)
        entries = np.column_stack((entry_rows, entry_steps))

        for entry in entries:
            row, step = entry
            self.stepRunner.entries_grid[row][step] += 2

        entries = []
        for row in range(self.stepRunner.nb_rows):
            for step in range(self.stepRunner.nb_steps[row]):
                self.stepRunner.entries_grid[row][step] -= 1
                self.stepRunner.entries_grid[row][step] = max(self.stepRunner.entries_grid[row][step], 0)
                self.stepRunner.entries_grid[row][step] = min(self.stepRunner.entries_grid[row][step], self.stepRunner.entries_max_hit)
                if self.stepRunner.entries_grid[row][step] >= self.stepRunner.entries_max_hit-3:
                    entries.append([row, step])

        if len(entries) == 0:
            return [], frame_warped

        rectangles = self.stepRunner.get_entries_rectangles(entries)
        cv.polylines(frame_warped, rectangles, True, (255, 0, 0), thickness=3)
        return entries, frame_warped


    def process_frame(self, frame):
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
        aruco_param = cv.aruco.DetectorParameters_create()

        # detect area corner on the screen
        corners_screen, corners_world = self.detect_screen_corners(
            frame, aruco_dict, aruco_param, self.ar_ids)
        if corners_screen is None:
            return None

        # draw grid on frame to make sure the area is well detected
        frame_feedback = self.draw_screen_feedback(
            frame, corners_screen, corners_world, self.world_w, self.world_h)

        # world to screen homography
        w_from_s, status = cv.findHomography(corners_screen, corners_world)
        self.w_from_s = self.w_from_s*0.9 + 0.1*w_from_s
        # Warp source image to destination based on homography
        margin = 60
        frame_warped = cv.warpPerspective(frame, self.w_from_s, (self.world_w, self.world_h))

        frame_warped, centers = PaperStepSequencer.detect_coins(frame_warped, self.marker_size)
        if len(centers) > 0:
            self.stepRunner.entries, frame_warped = self.get_grid_inputs(centers, frame_warped)

        # TODO: update sequencer inputs on most recent available image
        self.stepRunner.update_ar_content()
        ar = self.stepRunner.ar_content
        frame_warped[:, :, 0][ar > 0] = 0
        frame_warped[:, :, 1][ar > 0] = 0
        frame_warped[:, :, 2][ar > 0] = 255
        # cv.imshow("warped.png", frame_warped)
        # cv.waitKey(0)
        # cv.imwrite("output/warped.png", frame_warped)

        return frame_feedback, frame_warped

    def run(self, online=True):
        if online:
            cam = cv.VideoCapture(0)
            ret, frame = cam.read()

        cv.namedWindow("Camera")
        cv.namedWindow("Warped")

        img_counter = 0

        update_requested = False
        frame_is_valid = True

        c = -1
        while True:
            c = (c+1)%100
            if online:
                ret, frame = cam.read()
                if not ret:
                    break
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            else:
                frame = cv.imread(f"frames/opencv_frame_{c%3}.png", 0)

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


    cv_process = Thread(target=pss.run, kwargs={"online": False})
    cv_process.start()

    midi_process = Thread(target=pss.stepRunner.run)
    midi_process.start()

    midi_process.join()
    cv_process.join()

if __name__ == '__main__':
    main()

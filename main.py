import cv2 as cv
import numpy as np
# from PIL import Image
import os

def generate_markers(ar_ids):
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


def detect_screen_corners(frame, aruco_dict, ar_ids):
    # Initialize the detector parameters using default values
    parameters = cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # filter out unknown markers
    inds = [i for i, mid in enumerate(markerIds) if mid[0] in ar_ids]
    markerCorners = [markerCorners[ind] for ind in inds]
    markerIds = [markerIds[ind] for ind in inds]

    # markerCorners, markerIds = detect_markers(frame, aruco_dict, ar_ids)
    markerCorners = [c[0] for c in markerCorners]
    markerIds = [i[0] for i in markerIds]
    corners_screen = []
    for i, ar_id in enumerate(ar_ids):
        ind = markerIds.index(ar_id)
        corners_screen.append(markerCorners[ind][i])
    corners_screen = np.array(corners_screen)
    return corners_screen


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

    cv.imshow("screen_feedback.png", frame)
    cv.waitKey(0)
    cv.imwrite("output/screen_feedback.png", frame)


def main():
    # load predefined dictionary
    ar_ids = [11, 22, 33, 44]
    aruco_dict = generate_markers(ar_ids)

    # get src image and store point source coordinate system
    # frame = np.asarray(Image.open("frames/frame0.jpg").convert('L'))
    frame = cv.imread("frames/frame0.jpg", 0)

    # detect area corner on the screen
    corners_screen = detect_screen_corners(frame, aruco_dict, ar_ids)
    W, H = 710, 480
    corners_world = np.array([[0, 0], [W, 0], [W, H], [0, H]])

    # drwa grid on frame to make sure the area is well detected
    draw_screen_feedback(frame, corners_screen, corners_world, W, H)

    # world to screen homography
    w_from_s, status = cv.findHomography(corners_screen, corners_world)
    # Warp source image to destination based on homography
    frame_warped = cv.warpPerspective(frame, w_from_s, (W, H))
    cv.imshow("warped.png", frame_warped)
    cv.waitKey(0)
    cv.imwrite("output/warped.png", frame_warped)

    print("done")

if __name__ == '__main__':
    main()
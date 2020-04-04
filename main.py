import cv2 as cv
import numpy as np
from PIL import Image

def generate_markers(dictionary, ar_ids):
    for ar_id in ar_ids:
        # Generate the marker
        markerImage = np.zeros((200, 200), dtype=np.uint8)
        cv.aruco.drawMarker(dictionary, ar_id, 200, markerImage, 1)

        cv.imwrite(f"markers/marker{ar_id}.png", markerImage)


def detect_markers(frame, dictionary, ar_ids):
    # Initialize the detector parameters using default values
    parameters = cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    # filter out unknown markers
    inds = [i for i, mid in enumerate(markerIds) if mid[0] in ar_ids]
    markerCorners = [markerCorners[ind] for ind in inds]
    markerIds = [markerIds[ind] for ind in inds]

    print("done detect_markers")

    return markerCorners, markerIds

def main():
    # load predefined dictionary
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

    ar_ids = [11, 22, 33, 44]
    generate_markers(dictionary, ar_ids)

    # get src image and store point source coordinate system
    frame = np.asarray(Image.open("frames/WIN_20200404_16_48_43_Pro.jpg").convert('L'))
    # fh, fw = frame.shape
    # pts_src = np.asarray([[0,0], [fw, 0], [fw, fh], [0, fh]])

    # frame = cv.imread("frames/WIN_20200404_09_50_49_Pro_multiple.jpg", 0)

    corners, ids = detect_markers(frame, dictionary, ar_ids)
    corners = [c[0] for c in corners]
    ids = [i[0] for i in ids]
    corners_screen = []
    for i, ar_id in enumerate(ar_ids):
        ind = ids.index(ar_id)
        print(ind, ar_id, i)
        print( corners[ind])
        corners_screen.append(corners[ind][i])
    corners_screen = np.array(corners_screen)

    W, H = 710, 480

    corners_world = np.array([[0, 0], [710, 0], [710, 480], [0, 480]])

    # world to screen homography
    w_from_s, status = cv.findHomography(corners_screen, corners_world)
    s_from_w, status = cv.findHomography(corners_world, corners_screen)

    # Warp source image to destination based on homography
    warped_image = cv.warpPerspective(frame, w_from_s, (710, 480))
    cv.imwrite("warped.png", warped_image)


    lines = []
    for x in range(0, W+1, 10):
        pts_world = np.array([[x, 0, 1], [x, H, 1]])

        pts_screen = np.dot(s_from_w, pts_world.transpose()).transpose()
        pts_screen = pts_screen/pts_screen[:, 2:3]
        pts_screen = pts_screen[:, :2]

        line = np.array(pts_screen, dtype=np.int32)
        lines.append(line)
    for y in range(0, H+1, 10):
        pts_world = np.array([[0, y, 1], [W, y, 1]])

        pts_screen = np.dot(s_from_w, pts_world.transpose()).transpose()
        pts_screen = pts_screen/pts_screen[:, 2:3]
        pts_screen = pts_screen[:, :2]

        line = np.array(pts_screen, dtype=np.int32)
        lines.append(line)

    frame = np.stack([frame, frame, frame], axis=2)
    # cv.polylines(frame, [pts_screen], True, (0,0,255))
    cv.polylines(frame, lines, True, (0,255,0))


    # cv.polylines(frame, plx, True, (0,0,255))
    # cv.polylines(frame, ply, True, (0,255,0))

    cv.imwrite("out.png", frame)


    print("done main")

if __name__ == '__main__':
    main()
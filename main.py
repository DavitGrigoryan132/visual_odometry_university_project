import cv2
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

from visualizer import SLAMVisualizer


def detect_and_describe(image: np.ndarray):
    """
    Detect and extract feature descriptors using ORB.
    """
    orb = cv2.ORB().create()
    mask: np.ndarray = None
    keypoints, descriptors = orb.detectAndCompute(image, mask)
    return keypoints, descriptors


def match_features(desc1, desc2):
    """
    Match features using a brute force matcher.
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def estimate_motion(points1, points2, camera_matrix):
    """
    Estimate camera motion from a pair of subsequent images.
    """

    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, camera_matrix)

    return R, t, mask


def draw_custom_matches(img1, points1, img2, points2, matches):
    """
    Draw matches between two images using lists of keypoints.

    :param img1: First image.
    :param points1: List of keypoints in the first image as [[x1, y1], [x2, y2], ...].
    :param img2: Second image.
    :param points2: List of keypoints in the second image as [[x1, y1], [x2, y2], ...].
    :param matches: List of tuples representing matches as [(index1, index2), ...].
    """
    # Convert list of points into list of cv2.KeyPoint objects
    kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in points1]
    kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in points2]

    # Convert match tuples into cv2.DMatch objects
    dmatches = [cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _imgIdx=0, _distance=0) for idx1, idx2 in matches]

    # Draw matches
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, dmatches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    output_image = cv2.resize(output_image, None, fx=0.5, fy=0.5)
    # Convert BGR to RGB for matplotlib since OpenCV uses BGR by default
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return output_image


def draw_matches(img1, img2, kp1, kp2, matches):
    """
    Draws lines between matching keypoints of two images.
    """
    # Draw matches
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the image
    cv2.imshow("Matches", result)
    key = cv2.waitKey(0)
    if ord("q") == key:
        cv2.destroyAllWindows()
        exit(0)

    cv2.destroyAllWindows()


def filter_matches_with_ransac(kp1, kp2, matches, camera_matrix):
    """
    Filter matches using the RANSAC algorithm to find the essential matrix.

    :param kp1: Keypoints from the first image.
    :param kp2: Keypoints from the second image.
    :param matches: Raw matches between kp1 and kp2.
    :param camera_matrix: The camera intrinsic matrix.
    :return: Essential matrix, mask of inliers, and filtered matches.
    """
    # Convert keypoints to an array of points
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Compute the essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask = mask.ravel().tolist()

    # Filter matches according to the RANSAC results
    filtered_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    return E, mask, filtered_matches


def plot_positions_and_rotations(positions, rotations):
    """
    Plots 3D positions and orientations.

    :param positions: A list of (x, y, z) tuples representing the translations.
    :param rotations: A list of 3x3 numpy arrays representing the rotation matrices.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory of positions
    xs, ys, zs = zip(*positions)
    # ax.plot(xs, ys, zs, 'o-', label='Trajectory')

    # Scale factor for displaying the rotation axes
    scale = np.max(np.abs(np.array(positions))) / 10

    # Colors for the axes
    colors = ['r', 'g', 'b']  # Red for X, Green for Y, Blue for Z

    # Plot each rotation matrix as three lines
    for pos, R in zip(positions, rotations):
        # Start point of each line is the position
        start_point = np.array(pos)

        # End point is the position plus the column of the rotation matrix scaled
        for i in range(3):
            end_point = start_point + R[:, i] * scale
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]], colors[i] + '-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('3D Positions and Rotations')
    plt.show()


fx = 300
fy = 400
cx = 720 // 2
cy = 1280 // 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Define camera matrix
prev_image = None
R_total = np.eye(3)
t_total = np.zeros((3, 1))
image_paths = natsorted(glob.glob("data"))

rotations = []
translations = []
images = []

for img_path in tqdm.tqdm(image_paths):
    curr_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if prev_image is not None:
        kp1, desc1 = detect_and_describe(prev_image)
        kp2, desc2 = detect_and_describe(curr_image)
        matches = match_features(desc1, desc2)
        _, mask, matches = filter_matches_with_ransac(kp1, kp2, matches, camera_matrix)

        points1 = []
        points2 = []

        for i, m in enumerate(matches):
            if mask[i]:
                points1.append(kp1[m.queryIdx].pt)
                points2.append(kp2[m.trainIdx].pt)

        points1 = np.asarray(points1)
        points2 = np.asarray(points2)

        images.append(draw_custom_matches(prev_image, points1, curr_image, points2,
                                          np.asarray([range(len(points1)), range(len(points1))]).T))
        R, t, mask = estimate_motion(points1, points2, camera_matrix)
        R_total = R @ R_total
        t_total += t
        rotations.append(R_total)
        translations.append(t_total.T[0].copy())

    prev_image = curr_image

visualizer = SLAMVisualizer(translations, images)
visualizer.run()

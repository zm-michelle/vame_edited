from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2 as cv
import os
import tqdm
from scipy.ndimage import median_filter

from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


def consecutive(
    data: np.ndarray,
    stepsize: int = 1,
) -> List[np.ndarray]:
    """
    Find consecutive sequences in the data array.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    stepsize : int, optional
        Step size. Defaults to 1.

    Returns
    -------
    List[np.ndarray]
        List of consecutive sequences.
    """
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def nan_helper(y: np.ndarray) -> Tuple:
    """
    Identifies indices of NaN values in an array and provides a function to convert them to non-NaN indices.

    Parameters
    ----------
    y : np.ndarray
        Input array containing NaN values.

    Returns
    -------
    Tuple[np.ndarray, Union[np.ndarray, None]]
        A tuple containing two elements:
        - An array of boolean values indicating the positions of NaN values.
        - A lambda function to convert NaN indices to non-NaN indices.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


# def interpol_all_nans(arr: np.ndarray) -> np.ndarray:
#     """
#     Interpolates all NaN values in the given array.

#     Parameters
#     ---------
#     arr : np.ndarray
#         Input array containing NaN values.

#     Returns
#     -------
#     np.ndarray
#         Array with NaN values replaced by interpolated values.
#     """
#     y = np.transpose(arr)
#     nans, x = nan_helper(y)
#     y[nans] = np.interp(x(nans), x(~nans), y[~nans])
#     arr = np.transpose(y)
#     return arr


def interpol_first_rows_nans(arr: np.ndarray) -> np.ndarray:
    """
    Interpolates NaN values in the given array.

    Parameters
    ----------
    arr : np.ndarray
        Input array with NaN values.

    Returns
    -------
    np.ndarray
        Array with interpolated NaN values.
    """
    y = np.transpose(arr)
    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])
    arr = np.transpose(y)
    return arr


def interpolate_nans_with_pandas(data: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values along the time axis of a 3D NumPy array using Pandas.

    Parameters
    ----------
    data : numpy.ndarray
        Input 3D array of shape (time, keypoints, space).

    Returns
    -------
    numpy.ndarray:
        Array with NaN values interpolated.
    """
    for kp in range(data.shape[1]):  # Loop over keypoints dimension
        for sp in range(data.shape[2]):  # Loop over space dimension (x, y)
            series = pd.Series(data[:, kp, sp])
            series_interpolated = series.interpolate(
                method="linear",
                limit_direction="both",
                axis=0,
            )
            data[:, kp, sp] = series_interpolated.values
    return data


def crop_and_flip_legacy(
    rect: Tuple,
    src: np.ndarray,
    points: List[np.ndarray],
    ref_index: Tuple[int, int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Crop and flip the image based on the given rectangle and points.

    Parameters
    ----------
    rect : Tuple
        Rectangle coordinates (center, size, theta).
    src: np.ndarray
        Source image.
    points : List[np.ndarray]
        List of points.
    ref_index : Tuple[int, int]
        Reference indices for alignment.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        Cropped and flipped image, and shifted points.
    """
    # Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)

    # shift DLC points
    x_diff = center[0] - size[0] // 2
    y_diff = center[1] - size[1] // 2
    dlc_points_shifted = []
    for i in points:
        point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
        point[0] -= x_diff
        point[1] -= y_diff
        dlc_points_shifted.append(point)

    # Perform rotation on src image
    dst = cv.warpAffine(src.astype("float32"), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)

    # check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = (
            (size[0] // 2, size[0] // 2),
            size,
            180,
        )  # should second value be size[1]? Is this relevant to the flip? 3/5/24 KKL
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # Get rotation matrix
        M = cv.getRotationMatrix2D(center, theta, 1)

        # shift DLC points
        x_diff = center[0] - size[0] // 2
        y_diff = center[1] - size[1] // 2

        points = dlc_points_shifted
        dlc_points_shifted = []

        for i in points:
            point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
            point[0] -= x_diff
            point[1] -= y_diff
            dlc_points_shifted.append(point)

        # Perform rotation on src image
        dst = cv.warpAffine(out.astype("float32"), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)
    return out, dlc_points_shifted


def background(
    project_path: str,
    session: str,
    video_path: str,
    num_frames: int = 1000,
    save_background: bool = True,
) -> np.ndarray:
    """
    Compute background image from fixed camera.

    Parameters
    ----------
    project_path : str
        Path to the project directory.
    session : str
        Name of the session.
    video_path : str
        Path to the video file.
    num_frames : int, optional
        Number of frames to use for background computation. Defaults to 1000.

    Returns
    -------
    np.ndarray
        Background image.
    """
    logger.info("Computing background image ...")

    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise Exception(f"Unable to open video file: {video_path}")

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    height, width, _ = frame.shape
    frames = np.zeros((height, width, num_frames))

    for i in tqdm.tqdm(
        range(num_frames),
        disable=not True,
        desc="Compute background image for session %s" % session,
    ):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1, rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[..., i] = gray

    logger.info("Finishing up!")
    medFrame = np.median(frames, 2)
    background = median_filter(medFrame, (5, 5))

    if save_background:
        np.save(
            os.path.join(
                project_path,
                "data",
                "processed",
                session + "-background.npy",
            ),
            background,
        )

    capture.release()
    return background

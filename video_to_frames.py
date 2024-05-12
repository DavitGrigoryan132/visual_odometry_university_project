import cv2
import os
import argparse


def save_frames(video_path, save_dir):
    """
    Reads a video from the specified path and saves all frames as images in the specified directory.

    :param video_path: Path to the video file.
    :param save_dir: Directory to save the frames.
    """
    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read

        # Save frame as JPEG file
        frame_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved {frame_path}")
        frame_idx += 1

    # Release the video capture object
    cap.release()
    print("Released video resource.")


parser = argparse.ArgumentParser(description="Saves frames of video")
parser.add_argument("--video", required=True, type=str, help="Source video path")
parser.add_argument("--output", default="data", type=str, help="Destination directory where must be saved video frames")

args = parser.parse_args()
video_path = args.video
save_dir = args.output
save_frames(video_path, save_dir)


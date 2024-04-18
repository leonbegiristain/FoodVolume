import cv2
import os
import sys
from exif import Image


def extract_frames(video_path: str, reference_path: str, output_folder: str, n_frames: int = 10) -> None:
    """
    Video frame extractor. Extracts n_frames from the video located 
    at video_path and saves them as .jpg files in the output_folder.
    It also copies the EXIF data from the reference image located at
    reference_path to the extracted frames.
    """


    # Get the focal length, company and model from the reference image
    with open(reference_path, "rb") as f:
        img = Image(f)
        focal = img.focal_length
        make = img.make
        model = img.model

    # Clean output folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            if not file.endswith(".mp4"):
                os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    take_every = length // n_frames

    for i in range(take_every * n_frames):

        ret, frame = cap.read()

        if not ret:
            break

        
        if i % (take_every) == 0:
            # Save the frame as an image
            frame_path = os.path.join(output_folder, f"frame_{i//take_every:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            # Copy the EXIF data from the reference image to the frame
            with open(frame_path, "rb") as f:
                img = Image(f)
                img.make = make
                img.model = model
                img.focal_length = focal

            # Save the frame with the EXIF data
            with open(frame_path, "wb") as f:
                f.write(img.get_file())


    print(f"{n_frames} frames saved in: {output_folder}")
    cap.release()



if __name__ == "__main__":
    name = sys.argv[1]
    video_path = f"./Images/{name}/{name}.mp4"
    reference_path = "./potato.jpg"
    output_folder = "./Images/" + name

    extract_frames(video_path, reference_path, output_folder, n_frames = int(sys.argv[2]))


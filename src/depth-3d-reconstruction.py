import torch
from Object_Reconstruction import preprocess_pcd
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import os
import piexif
import sys
metric3d_path = './Metric3D'
sys.path.append(metric3d_path) # Add Metric3D to path, otherwise imports inside Metric3D will fail
from Metric3D.predict_dir import predict_folder
from FastSAM.fastsam import FastSAM, FastSAMPrompt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

source_path = './image-source/banana'
input_type = 'jpg'

rec_input_path = './input/banana'
rec_input_name = 'image'

assert os.path.exists(source_path), "Input path does not exist"
source_path = os.path.abspath(source_path)

# 0. Get intrinsic matrix

print('========================')
print('Getting intrinsic matrix')
print('========================')

# first_file = next(file for file in os.listdir(source_path) if file.endswith('.' + input_type))
# img = Image.open(os.path.join(source_path, first_file))
# exif_data = img._getexif()
# for tag_id in exif_data:
#     # get the tag name, instead of human unreadable tag id
#     tag = TAGS.get(tag_id, tag_id)
#     data = exif_data.get(tag_id)
#     # decode bytes 
#     if isinstance(data, bytes):
#         data = data.decode()
#     print(f"{tag:25}, {tag_id}: {data}")

# print(exif_data)
# focal_length = exif_data[37386] # Focal Length
# print(f"Focal Length: {focal_length}")
# if exif_data.get(256) is None or exif_data.get(257) is None:
#     print("Image Width and Height not found in metadata")
#     width, height = img.size
# else:
#     width, height = exif_data[256], exif_data[257] # Image Width, Image Height
# print(f"Image Width: {width}, Image Height: {height}")

# sensor_pixel_size = 1.2 # um
# focal_length_pixel = focal_length / (sensor_pixel_size * 1e-3) # mm to um --> pixel

# K = np.array([
#     [focal_length_pixel, 0,                  width / 2  ],
#     [0,                  focal_length_pixel, height / 2 ],
#     [0,                  0,                  1          ]
# ], dtype=np.float64)

# print(K)

# 1. Copy RGB images

print('========================')
print('Copying RGB images and resizing')
print('========================')

print('Deleting existing files')
if os.path.exists(os.path.join(rec_input_path, 'rgb')):
    for file in os.listdir(os.path.join(rec_input_path, 'rgb')):
        os.remove(os.path.join(rec_input_path, 'rgb', file))
else:
    os.makedirs(os.path.join(rec_input_path, 'rgb'))
if os.path.exists(os.path.join(rec_input_path, 'depth')):
    for file in os.listdir(os.path.join(rec_input_path, 'depth')):
        os.remove(os.path.join(rec_input_path, 'depth', file))
else:
    os.makedirs(os.path.join(rec_input_path, 'depth'))

# Assuming all images have the same size

for count, file in enumerate(os.listdir(source_path)):
    if file.endswith('.' + input_type):
        # Copy RGB images
        img = Image.open(os.path.join(source_path, file))
        # img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        name_count = str(count + 1).rjust(3, '0')
        img.save(os.path.join(rec_input_path, 'rgb', f'{rec_input_name}{name_count}.{input_type}'))

# 2. Metric 3D depth estimation

print('========================')
print('Running Metric 3D')
print('========================')


# predict_folder(
#     input_dir=os.path.join(rec_input_path, 'rgb'),
#     output_dir=os.path.join(rec_input_path, 'depth'),
#     fx=focal_length_pixel,
#     cx=width / 2,
#     cy=height / 2,
#     metric3d_path=metric3d_path,
#     # config='./mono/configs/HourglassDecoder/vit.raft5.large.py',
#     # ckpt='./weight/metric_depth_vit_large_800k.pth',
#     # device=device,
# )

# 3. FastSAM mask prediction

print('========================')
print('Running FastSAM')
print('========================')

import matplotlib.pyplot as plt

if os.path.exists(os.path.join(rec_input_path, 'mask')):
    for file in os.listdir(os.path.join(rec_input_path, 'mask')):
        os.remove(os.path.join(rec_input_path, 'mask', file))
        # plt.imshow(np.load(os.path.join(rec_input_path, 'mask', file)))
        # plt.show()
else:
    os.makedirs(os.path.join(rec_input_path, 'mask'))

model = FastSAM('../FastSAM-s.pt')

import cv2
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    global firstX, firstY, secondX, secondY, drawing, mode, img, isReady

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        firstX, firstY = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = np.copy(original_img)
            cv2.rectangle(img, (firstX, firstY), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        secondX, secondY = x, y
        cv2.rectangle(img, (firstX, firstY), (x, y), (0, 255, 0), 2)
        isReady = True
        print("Top-left coordinates:", firstX, firstY)
        print("Bottom-right coordinates:", x, y)


for count, file in enumerate(os.listdir(os.path.join(rec_input_path, 'rgb'))):
    print(f'Processing file number {count + 1}:', file)
    file_path = os.path.join(rec_input_path, 'rgb', file)
    everything_results = model(file_path, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(file_path, everything_results, device='cpu')

    # Load an image
    original_img = cv2.imread(file_path)
    original_img = cv2.resize(original_img, (original_img.shape[1]//2, original_img.shape[0]//2))

    img = np.copy(original_img)

    drawing = False  # True if mouse is pressed
    isReady = False
    firstX, firstY, secondX, secondY = -1, -1, -1, -1

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & (isReady or 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()

    firstX, firstY = firstX * 2, firstY * 2
    secondX, secondY = secondX * 2, secondY * 2

    print(firstX, firstY, secondX, secondY)

    mask = prompt_process.box_prompt(bboxes=[[firstX, firstY, secondX, secondY]])
    mask = np.array(mask)[0, :, :].astype(np.uint8)

    plt.imshow(mask)
    plt.plot(firstX, firstY, 'ro')
    plt.plot(secondX, secondY, 'ro')
    plt.show()
    
    # Make the mask smaller. This will help removing the jagged edges of the depth map
    # Find the boundary pixels
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask)  # Create a blank image for the boundary
    cv2.drawContours(boundary, contours, -1, 255, 1)  # Draw boundary with a thickness of 1 pixel

    # Dilate the boundary (adjust thickness as needed)
    kernel = np.ones((10,10), np.uint8)  # A 5x5 dilation kernel
    dilated_boundary = cv2.dilate(boundary, kernel, iterations=1)

    # Modify the original mask
    mask[dilated_boundary == 255] = 0

    name_count = str(count + 1).rjust(3, '0')
    np.save(os.path.join(rec_input_path, 'mask', f'{rec_input_name}{name_count}.npy'), mask)
    

# 4. RGBD to PCD and clean up

# print(K)
# preprocess_pcd.rgbd_to_pcd(rec_input_path, './output/salt', rec_input_name, debug=True, input_type=input_type, K=K)

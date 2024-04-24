
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import cv2
import numpy as np

def segment_view(model: FastSAM, file_path: str):    
    firstX, firstY, secondX, secondY = -1, -1, -1, -1
    isReady = False
    drawing = False

    def draw_rectangle(event, x, y, flags, param):
        nonlocal firstX, firstY, secondX, secondY, isReady, img, original_img, drawing

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

    everything_results = model(file_path, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(file_path, everything_results, device='cpu')

    # Load an image
    original_img = cv2.imread(file_path)
    original_img = cv2.resize(original_img, (original_img.shape[1]//2, original_img.shape[0]//2))

    img = np.copy(original_img)

    isReady = False

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & (isReady or 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()

    firstX, firstY = firstX * 2, firstY * 2
    secondX, secondY = secondX * 2, secondY * 2

    if firstX > secondX:
        firstX, secondX = secondX, firstX
    if firstY > secondY:
        firstY, secondY = secondY, firstY

    mask = prompt_process.box_prompt(bboxes=[[firstX, firstY, secondX, secondY]])
    mask = np.array(mask)[0, :, :].astype(np.uint8)
    
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

    return mask
# 3D Reconstruction, Segmentation and Food Volume Estimation from Monocular Video
* León Begiristain Ribó, Faculty of Physics and Astronomy
* Sámuel Szajbély, Faculty of Physics and Astronomy
* Arjan Siddhpura, Faculty of Mathematics and Computer Science

We outline a comprehensive approach for 3D reconstruction, segmentation, and volume estimation of objects from monocular video footage. The paper and this code addresses the challenges of featureless environments and view selection, proposing a multi-step pipeline that leverages state-of-the-art techniques in computer vision and machine learning. Key components include utilizing openMVG and openMVS libraries for 3D reconstruction, developing a novel algorithm for view selection based on the Maximum Diversity Problem (MDP), and employing FastSAM for accurate segmentation. The method also incorporates PoissonRecon for mesh reconstruction and volume computation, with scaling to metric units using a reference object. Experimental results demonstrate the effectiveness of the approach on various food items, while discussing potential extensions and improvements for future research.  

--- 

The following summarizes the functioning of the individual files of the source code.

* `extract_frames.py`  
This Python script extracts frames from a video file, saves them as JPEG images, and copies EXIF data from a reference image to the extracted frames. The number of frames to extract is specified by the user.

* `read_dmap.py`  
  The Python function in this script, loadDMAP, reads a depth map file (.dmap) and parses its content into a dictionary format. It first opens the file and reads various metadata such as file type, content type, image dimensions, depth map dimensions, and other parameters. Based on the content type, it determines if the depth map contains normal information, confidence values, or multiple views. It then extracts camera intrinsics (K), rotation (R), translation (C), and the depth map itself. Additionally, it parses normal maps, confidence maps, and views maps if available. Finally, it returns a dictionary containing all parsed information, enabling easy access to the depth map data and associated metadata.

* `segment_view.py`  
  This code defines a function segment_view that segments a specific view within an image using the FastSAM model. It first initializes variables for drawing a rectangle on the image to select the desired view. Then, it utilizes the FastSAM model to predict object masks for the entire image. After the user selects a region of interest by drawing a rectangle, the function extracts the selected area and refines it to remove jagged edges caused by the initial segmentation. Finally, it returns the refined mask representing the segmented view.

* `select_views.py`  
  This script defines a function select_views that selects a subset of views from a list of depth maps based on the maximum minimum distance to selected elements. It uses a greedy algorithm to iteratively select views that maximize the minimum distance to previously selected views. The distance metric is a combination of Euclidean distance between view positions and the number of points in the depth maps. The code also includes helper functions for calculating distances, computing the number of points in a depth map, and normalizing numerical values. In the example usage, random depth maps are generated, and the function is called to select a subset of views based on specified parameters. Finally, a plot is generated to visualize the selected views alongside all available views.

---

### How to run

We have only tested the pipeline on windows and all the binaries are for windows.

How to prepare data to run the code:
 - If you want to run the code using a video, the video should be placed in the ./videos/ folder. If running with images, they should be placed on ./images/.
 - If running with video, a reference image using the phone that took the video should be placed on ./reference-images/ and the "reference_path" variable should be adjusted accordingly.
 - The company, model name and sensor width should be written in ./sensor_width_database/sensor_width_camera_database.txt as written on the reference image metadata.






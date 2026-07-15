# Computer Vision Projects

A collection of computer vision projects implemented in various languages

Check out the project overviews below.

## Detecting Faces with SVMs [⤴](./face_detection_with_SVMs)

A simple face detector that uses a Support Vector Machine (SVM) trained on Histograms of Oriented Gradients (HOG).

The HOG pattern learned by the SVM:

![HOG pattern learned by the SVM](./face_detection_with_SVMs/code/visualizations/hog_template.png)

Face detection example:

![face detection example](./face_detection_with_SVMs/code/visualizations/detections_madaboutyou.jpg.png)

## Efficient Graph-Based Image Segmentation [⤴](./graph_based_image_segmentation)

A fast Python implementation of the classic image segmentation algorithm by Felzenszwalb and Huttenlocher.

Examples:

- beach.jpg:

![beach](./graph_based_image_segmentation/gallery/beach.jpg)
![beach segmentation](./graph_based_image_segmentation/gallery/beach_random.jpg)

- Ioannina clock tower:

<img src="./graph_based_image_segmentation/gallery/ioannina_clock_tower.jpg" width="400"/> <img src="./graph_based_image_segmentation/gallery/ioannina_clock_tower_random.jpg" width="400"/>

## Seam Carving for Content-Aware Image Resizing [⤴](./seam_carving_octave)

An Octave implementation of the classic image resizing algorithm.

The vertical/horizontal seams with the lowest energy are iteratively removed until the target width/height is reached.

| Image | Energy | Minimum Energy Seams |
| :---: | :---: | :---: |
|<img width="240" height="180" alt="lake_h200_linfty" src="https://github.com/user-attachments/assets/bf89e373-cae1-4fcd-9e34-3f3b6260ecdc" />|<img width="240" height="180" alt="lake_h200_linfty_energy" src="https://github.com/user-attachments/assets/eecf7ec3-26a2-40f6-8e9e-7a9985999686" />|<img width="240" height="180" alt="lake_h200_linfty_minEnergy" src="https://github.com/user-attachments/assets/417f5c57-9cc5-495f-92b8-aa24bdb1beff" />|
|<img width="240" height="180" alt="lake_v200_linfty" src="https://github.com/user-attachments/assets/79ad0f3e-f0ef-41d8-8894-5188b89ddd4b" />|<img width="240" height="180" alt="lake_v200_linfty_energy" src="https://github.com/user-attachments/assets/4be839f0-b5a7-4dee-b7c3-d0f2fc7471b7" />|<img width="240" height="180" alt="lake_v200_linfty_minEnergy" src="https://github.com/user-attachments/assets/cbe67919-5ad6-4f15-8f8e-f96de46c2001" />|

## Histogram Equalization [⤴](./histogram_equalization)

An implementation of the contrast enhancement technique known as histogram equalization.

![lena, it's histogram, equalized lena and it's histogram](./histogram_equalization/lena_report.png)

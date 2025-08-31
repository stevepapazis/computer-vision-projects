# Detecting Faces with SVMs

A simple face detector that uses a Support Vector Machine (SVM) trained on Histograms of Oriented Gradients (HOG).

The SVM was trained on a database of 6,713 cropped 36x36 faces (and their horizontal mirrorings) extracted from the Caltech Web Faces project. Negative examples were randomly sampled from a database of images without faces.

For each image, HOG were computed. When the image size exceeds 36x36, a sliding window approach is used. Training on these HOGs, the SVM learns to detect faces.

## The HOG pattern learned by the SVM

![HOG pattern learned by the SVM](./code/visualizations/hog_template.png)

## Although relative simple, the system exhibits surprisingly high accuracy

![average precision graph](./code/visualizations/average_precision.png)

## Some examples from ./code/visualizations

![detection example 1](./code/visualizations/detections_addams-family.jpg.png)

![detection example 2](./code/visualizations/detections_Brazil.jpg.png)

![detection example 3](./code/visualizations/detections_madaboutyou.jpg.png)

![detection example 4](./code/visualizations/detections_torrance.jpg.png)

![detection example 5](./code/visualizations/detections_tress-photo.jpg.png)

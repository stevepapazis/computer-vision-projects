# Computer Vision Projects

A collection of computer vision projects implemented in various languages

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

![](./graph_based_image_segmentation/input_images/beach.jpg)
![](./graph_based_image_segmentation/gallery/beach.jpg)

- Ioannina clock tower:

<img src="./graph_based_image_segmentation/input_images/ioannina_clock_tower.jpg" width="400"/> <img src="./graph_based_image_segmentation/gallery/ioannina_clock_tower.jpg" width="400"/>

## Seam Carving for Content-Aware Image Resizing [⤴](./seam_carving_octave)

An octave implementation of the classic image resizing algorithm.

The vertical/horizontal seams with the lowest energy are iteratively removed until the target width/height is reached.

<table>
    <tr>
        <td>
            <video width="480" autoplay loop muted>
            <source src="./seam_carving_octave/gallery/lake_h200_linfty.mp4" type="video/mp4">
            </video>
        </td>
        <td>
            <video width="480" autoplay loop muted>
            <source src="./seam_carving_octave/gallery/lake_h200_linfty_energy.mp4" type="video/mp4">
            </video>
        </td>
        <td>
            <video width="480" autoplay loop muted>
            <source src="./seam_carving_octave/gallery/lake_h200_linfty_minEnergy.mp4" type="video/mp4">
            </video>
        </td>
        </tr>
    <tr>
        <td>
            <video width="480" autoplay loop muted>
            <source src="./seam_carving_octave/gallery/lake_v200_linfty.mp4" type="video/mp4">
            </video>
        </td>
        <td>
            <video width="480" autoplay loop muted>
            <source src="./seam_carving_octave/gallery/lake_v200_linfty_energy.mp4" type="video/mp4">
            </video>
        </td>
        <td>
            <video width="480" autoplay loop muted>
            <source src="./seam_carving_octave/gallery/lake_v200_linfty_minEnergy.mp4" type="video/mp4">
            </video>
        </td>
    </tr>
</table>


## Histogram Equalization [⤴](./histogram_equalization)

An implementation of the contrast enhancement technique known as histogram equalization.

![lena, it's histogram, equalized lena and it's histogram](./histogram_equalization/lena_report.png)

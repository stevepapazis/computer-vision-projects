import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



class Histogram:
    """A class used to represent histograms of monochromatic images"""
    def __init__(self, image, name=None):
        self.name = name
        self.image = image
        self.compute_frequencies()

    def generate_from(file, name=None):
        """Generates the histogram of an image file"""
        try:
            return Histogram( plt.imread(file), name )
        except FileNotFoundError:
            print( 'Unable to locate the image at "', file, '".', sep='' )
            exit()

    def compute_frequencies(self):
        """Computes the frequencies of all the pixels in the image"""
        self.frequencies = np.zeros(256)
        size = np.prod(self.image.shape)
        frequencies = np.unique(self.image, return_counts=True)
        self.frequencies[ frequencies[0].astype(np.int8) ] = frequencies[1]/size
        
    def equalize(self):
        """Equalizes the histogram and transforms the image"""
        cdf = np.cumsum(self.frequencies)
        transformation = np.round(cdf*255)
        self.image = transformation[self.image]
        self.compute_frequencies()

    def equalize_and_print_report(self, save_to_file=None, show_plot=None):
        """Equalizes the image and prints the plots

        - The report is saved as an image when a file name is provided.
        - The result will not appear on the screen, unless matplotlib.pyplot.show
          is called or show_plot is set to True.
        """
        title = "Histogram equalization" + (":  " + self.name if self.name else "")
        fig = plt.figure(title, figsize=(7,7), layout="constrained")
        initial, equalized = fig.subfigures(nrows=2, ncols=1)

        self.print(initial, "Initial")
        self.equalize()
        self.print(equalized, "Equalized")

        if save_to_file: fig.savefig(save_to_file)

        if show_plot: plt.show()

    def print(self, subfigure, name):
        """Attach the image and the histogram to a subfigure"""
        image, histplt = subfigure.subplots(nrows=1, ncols=2)

        image.set_title(name + " image")
        img = image.imshow(self.image, cmap="gray", vmin=0, vmax=255)
        image.set_axis_off()
        plt.colorbar(img, ax=image, location="left")

        histplt.set_title(name + " histogram")
        histplt.bar( np.arange(256), self.frequencies, width=1 )
        histplt.set_xlabel("Grey scale")
        histplt.set_ylabel("Frequencies")
        histplt.set_xlim([-5,260])

    


if __name__ == "__main__":
    script_path = Path(__file__).parent
    resolve = lambda path: script_path.joinpath(path).resolve()

    lena = Histogram.generate_from( resolve("./2.bmp"), "lena" )
    lena.equalize_and_print_report( save_to_file = resolve("./lena_report.png") )

    peppers = Histogram.generate_from( resolve("./4.bmp"), "peppers" )
    peppers.equalize_and_print_report( save_to_file = resolve("peppers_report.png") )
    
    plt.show()

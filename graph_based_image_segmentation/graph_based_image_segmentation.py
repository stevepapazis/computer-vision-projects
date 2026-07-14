from __future__ import annotations
from typing import Callable

from collections import namedtuple

import numpy as np
from numba import njit
import skimage

import argparse
from pathlib import Path
from functools import partial
import time



# The classic disjoint-set (union-find) data structure
#   number_of_sets :   pointer to the number of disjoint sets
#   parents :          pointers to the parent nodes representing each set
#   sizes :            pointers to the set sizes; only the roots are guaranteed to contain the correct sizes
# The class is implemented as a namedtuple in order to utilize numba's njit decorator on its "methods"
DisjointSets = namedtuple("DisjointSets", ["number_of_sets", "parents", "sizes"])

def _create_DisjointSets(total_elements: int) -> DisjointSets:
    """Creates a DisjointSets instance with the given number of elements"""
    return DisjointSets(
        number_of_sets = np.uint64([total_elements]),
        parents        = np.arange(total_elements, dtype=np.uint64),
        sizes          = np.ones(total_elements, dtype=np.uint64)
    )

@njit("uint64(uint64, uint64[:])")
def _find_set(element: int, parents: np.uint64) -> int:
    """Returns the representative (root) of the set containing the given element"""
    while element != parents[element]:
        parents[element] = parents[parents[element]]
        element = parents[element]
    return element

@njit("void(uint64, uint64, uint64[:], uint64[:], uint64[:])")
def _join_sets(element1: int, element2: int, number_of_sets: np.uint64, parents: np.uint64, sizes: np.uint64) -> None:
    """Joins the two sets containing the given elements"""

    root1 = _find_set(element1, parents)
    root2 = _find_set(element2, parents)

    if root1 == root2: return

    if sizes[root1] < sizes[root2]:
        root1, root2 = root2, root1

    parents[root2] = root1
    sizes[root1] += sizes[root2]

    number_of_sets[0] -= 1

@njit("uint64(uint64, uint64[:], uint64[:])")
def _get_size(element: int, parents: np.uint64, sizes: np.uint64) -> int:
    """Returns the size of the component containing the given element"""
    root = _find_set(element, parents)
    return sizes[root]      # only the roots are guaranteed to have the correctly updated sizes

@njit("uint64[:](uint64[:], uint64[:])")
def _get_parents(number_of_sets: np.uint64, parents: np.uint64) -> np.uint64:
    """Returns the parents of each element"""
    num_of_unique_parents = np.unique(parents).size
    if number_of_sets[0] != num_of_unique_parents:
        for e in range(parents.size):
            parents[e] = _find_set(e, parents)
        number_of_sets[0] = num_of_unique_parents
    return parents


@njit
def _refine_segmentation(segmentation: DisjointSets, vertices: np.uint64, edges: np.uint64, weights: np.float64, k: int, m: int) -> DisjointSets:
    """Applies the Felzenszwalb-Huttenlocher algorithm to refine an initial segmentation of many small components.
    The vertices, edges and weights  must be sorted in ascending order of weight.
    k is a scale factor and M is the minimum acceptable component size."""

    max_edge_in_set_MST = np.zeros_like(vertices, dtype=np.float64)

    # Join similar sets according to the criteria of the algorithm
    for (px1, px2), weight in zip(edges.T, weights):
        c1 = _find_set(px1, segmentation.parents)   # An integer that represents the root of the component c1
        c2 = _find_set(px2, segmentation.parents)   # An integer that represents the root of the component c2

        if c1 == c2: continue

        tauC1 = k/_get_size(c1, segmentation.parents, segmentation.sizes)   # The threshold function τ(c1) of a component c1
        intC1 = max_edge_in_set_MST[c1]                                     # The internal difference Int(c2) of a component c1

        tauC2 = k/_get_size(c2, segmentation.parents, segmentation.sizes)   # The threshold function τ(c1) of a component c2
        intC2 = max_edge_in_set_MST[c2]                                     # The internal difference Int(c2) of a component c2

        min_intC1C2 = np.minimum(intC1 + tauC1, intC2 + tauC2)  # The minimum internal difference MInt(c1,c2) between two
                                                                # disconnected components c1 and c2

        if weight <= min_intC1C2:
            _join_sets(c1, c2, segmentation.number_of_sets, segmentation.parents, segmentation.sizes)
            max_edge_in_set_MST[c1] = max_edge_in_set_MST[c2] = weight

    # Join unacceptably small sets
    for (v1, v2) in edges.T:
        r1 = _find_set(v1, segmentation.parents)
        r2 = _find_set(v2, segmentation.parents)

        if r1 == r2: continue

        if _get_size(r1, segmentation.parents, segmentation.sizes) < m or _get_size(r2, segmentation.parents, segmentation.sizes) < m:
            _join_sets(r1, r2, segmentation.number_of_sets, segmentation.parents, segmentation.sizes)

    return segmentation



class Graph:
    def __init__(self, vertices = np.uint64([]), edges = np.uint64([]), weights = np.float64([])) -> None:
        self.vertices = vertices
        self.edges    = edges
        self.weights  = weights

    @staticmethod
    def create_grid_graph(image: np.float64, weighting_function: Callable[[np.float64], np.float64]) -> Graph:
        """Computes the weighted grid graph of a color image.

        Each pixel is connected to its north, south, east, and west neighbors.
        The edge weights, computed by the provided weighting function
        (e.g., Euclidean (L2) distance in color space), are stored in order of
        decreasing value: smaller weights indicate greater similarity."""

        height, width = image.shape[:2]
        vertices = np.arange(height*width, dtype=np.uint64).reshape(height, width)

        left_vertices   = vertices[:,:-1].reshape(-1)
        right_vertices  = vertices[:,1:].reshape(-1)
        top_vertices    = vertices[:-1,:].reshape(-1)
        bottom_vertices = vertices[1:,:].reshape(-1)

        # Get the pixels (rgb intensities) from the given indices
        get_pixels_from_indices = lambda image, region_indices: \
            image[  np.s_[ np.unravel_index(region_indices, image.shape[:2]) ]  ]

        left_vertex_values   = get_pixels_from_indices(image, left_vertices)
        right_vertex_values  = get_pixels_from_indices(image, right_vertices)
        top_vertex_values    = get_pixels_from_indices(image, top_vertices)
        bottom_vertex_values = get_pixels_from_indices(image, bottom_vertices)

        edges = np.hstack([
            np.stack([left_vertices, right_vertices]),
            np.stack([top_vertices, bottom_vertices]),
        ])

        distance = np.hstack([
            weighting_function(left_vertex_values, right_vertex_values),
            weighting_function(top_vertex_values, bottom_vertex_values),
        ])

        sorting = np.argsort(distance)

        return Graph(vertices.reshape(-1), edges[:,sorting], distance[sorting])

    def segment(self, scale_factor: int, minimal_acceptable_size: int = 50) -> DisjointSets:
        """Segments the graph into disjoint components as in:
        Felzenszwalb, P.F., Huttenlocher, D.P. Efficient Graph-Based Image Segmentation.
        International Journal of Computer Vision 59, 167-181 (2004).
        https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf"""

        segmentation = _create_DisjointSets(len(self.vertices))
        segmentation = _refine_segmentation(segmentation, self.vertices, self.edges, self.weights, scale_factor, minimal_acceptable_size)
        return segmentation


def compute_Lp_distance(pixels_in_region1: np.ndarray, pixels_in_region2: np.ndarray, p: float) -> np.ndarray:
    """Computes the pairwise Lp distance between the pixel values (RGB) of two image regions.
    For p>=1 and two pixels x and y, the Lp distance is defined as: (sum_i |x_i - y_i|^p)^(1/p)"""

    return np.sum(np.abs(pixels_in_region1-pixels_in_region2)**p, axis=1)**(1/p)

def compute_Linfinity_distance(pixels_in_region1: np.ndarray, pixels_in_region2: np.ndarray) -> np.ndarray:
    """Computes the pairwise L_infinity distance between the pixel values (RGB) of two image regions.
    For two pixels x and y, the L_infinity distance is defined as: max_i |x_i - y_i|"""

    return np.max(np.abs(pixels_in_region1-pixels_in_region2), axis=1)



VALID_PALETTE_OPTIONS = ["mean", "random"]


def visualize_segmentation(image: np.uint8, segmentation: DisjointSets, palette_to_use: str) -> np.uint8:
    """Colors the image based on the segmentation and the chosen color palette."""

    out_image = np.zeros_like(image, dtype=np.uint8)

    components = np.unique(_get_parents(segmentation.number_of_sets, segmentation.parents))

    for component in components:
        all_elements_in_set = np.argwhere(segmentation.parents == _find_set(component, segmentation.parents))
        indices = np.unravel_index(all_elements_in_set, shape=image.shape[:2])

        if palette_to_use == "mean":
            colour = np.mean(image[indices], axis=0)
        elif palette_to_use == "random":
            colour = np.random.randint(0, 255, size=3)
        else:
            raise ValueError(f"invalid value {palette_to_use=}; pick one of the following options: {VALID_PALETTE_OPTIONS}")

        out_image[indices] = colour

    return out_image




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""An implementation of the image segmentation algorithm introduced in:
        Felzenszwalb, P.F., Huttenlocher, D.P. Efficient Graph-Based Image Segmentation.
        International Journal of Computer Vision 59, 167-181 (2004).
        https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf"""
    )

    parser.add_argument(
        "--input", "-i",
        help="path to the input image",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output", "-o",
        help="path to the output image",
        type=Path
    )

    def _test_number_validity(condition):
        def test_condition(value):
            try:
                value = float(value)
                if condition(value):
                    return value
            except:
                pass
            raise argparse.ArgumentTypeError(f"{value} is an invalid input value")
        return test_condition

    parser.add_argument(
        "--scale-factor", "-k",
        help="positive scale constant controlling the trade-off between detail and region size (default: %(default)s)",
        type=_test_number_validity(lambda x: x>=0),
        default=300.0
    )

    parser.add_argument(
        "--sigma", "-s",
        help="positive constant employed in the pre-segmentation blurring (default: %(default)s)",
        type=_test_number_validity(lambda x: x>=0),
        default=0.8
    )

    parser.add_argument(
        "--minimal-acceptable-size", "-m",
        help="""positive value defining the minimum size (in pixels) of an acceptable component (default: %(default)s)""",
        type=_test_number_validity(lambda x: x>=0),
        default=50.0,
    )

    parser.add_argument(
        "--palette", "-p",
        help="color palette used in the output image (default: %(default)s)",
        default="random",
        choices=VALID_PALETTE_OPTIONS
    )

    parser.add_argument(
        "--weighting-function", "-l",
        help=f"""function measuring similarity between neighboring pixels;
            choose any float p>=1 to use Lp(x,y) = (sum_i |x_i - y_i|^p)^(1/p),
            or p=0 for L_infinity(x,y) = max_i |x_i - y_i| (default: %(default)s)""",
        type=_test_number_validity(lambda x: x>=1 or x==0),
        metavar="{p>=1 or p==0}",
        default=2.0,
    )

    parser.add_argument(
        "--branding", "-b",
        help="include the input parameters in the name of the output file",
        action="store_true"
    )

    parser.add_argument(
        "--verbose", "-v",
        help="display processing steps and timing information",
        action="store_true"
    )

    args = parser.parse_args()

    input_image = args.input
    output_image = args.output
    scale_factor = args.scale_factor
    sigma = args.sigma
    minimal_acceptable_size = args.minimal_acceptable_size
    palette = args.palette
    p = args.weighting_function
    lp_distance = compute_Linfinity_distance if p==0 else partial(compute_Lp_distance, p=p)
    branding = args.branding
    verbose = args.verbose

    parameters = [
        f"k={scale_factor}",
        f"s={sigma}",
        f"l{p}" if p!=0 else "linfty",
        f"m={minimal_acceptable_size}",
        palette
    ]

    if output_image is None:
        branding = True
        output_image = input_image

    if branding:
        name = output_image.stem
        suffix = output_image.suffix
        output_image = output_image.with_name(
            f"{name}_" + "_".join(parameters) + suffix
        )

    start = time.time()

    if verbose: print(f"Loading {input_image}...")

    image = skimage.io.imread(input_image)
    image = np.float64(image)
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)

    if verbose: print("Segmenting the image...")

    graph = Graph.create_grid_graph(blurred_image, lp_distance)
    segmentation = graph.segment(scale_factor, minimal_acceptable_size)
    segmented_image = visualize_segmentation(image, segmentation, palette)

    if verbose: print(f"Saving {output_image}...")

    skimage.io.imsave(output_image, segmented_image)

    if verbose: print(f"Finished in {time.time()-start} seconds")
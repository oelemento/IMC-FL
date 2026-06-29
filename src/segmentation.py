"""
Cell segmentation for Hyperion IMC data using Cellpose.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import warnings

# Suppress some warnings
warnings.filterwarnings('ignore', category=UserWarning)


def prepare_nuclear_image(
    image: np.ndarray,
    markers: list[str],
    nuclear_markers: list[str] = ['DNA1', 'DNA2'],
    percentile_norm: float = 99.5,
) -> np.ndarray:
    """Prepare nuclear image for segmentation by combining nuclear channels.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        nuclear_markers: markers to combine for nuclear signal
        percentile_norm: percentile for normalization

    Returns:
        2D normalized nuclear image
    """
    nuclear_channels = []

    for marker in nuclear_markers:
        if marker in markers:
            idx = markers.index(marker)
            channel = image[:, :, idx].astype(np.float32)
            nuclear_channels.append(channel)

    if not nuclear_channels:
        raise ValueError(f"No nuclear markers found. Available: {markers[:10]}...")

    # Combine channels (sum or max)
    if len(nuclear_channels) == 1:
        nuclear = nuclear_channels[0]
    else:
        # Use sum for better signal
        nuclear = np.sum(nuclear_channels, axis=0)

    # Normalize to 0-1
    vmin = np.percentile(nuclear, 100 - percentile_norm)
    vmax = np.percentile(nuclear, percentile_norm)
    if vmax > vmin:
        nuclear = (nuclear - vmin) / (vmax - vmin)
    nuclear = np.clip(nuclear, 0, 1)

    return nuclear


def segment_cells_watershed(
    nuclear_image: np.ndarray,
    min_distance: int = 10,
    threshold_method: str = 'otsu',
    min_size: int = 50,
) -> np.ndarray:
    """Fast watershed-based segmentation (runs in seconds on CPU).

    Args:
        nuclear_image: 2D normalized nuclear image (0-1)
        min_distance: minimum distance between cell centers
        threshold_method: 'otsu' or 'li'
        min_size: minimum cell size in pixels

    Returns:
        2D label mask
    """
    from scipy import ndimage
    from skimage import filters, morphology, segmentation, measure

    # Smooth
    smoothed = ndimage.gaussian_filter(nuclear_image, sigma=1)

    # Threshold
    if threshold_method == 'otsu':
        thresh = filters.threshold_otsu(smoothed)
    else:
        thresh = filters.threshold_li(smoothed)

    binary = smoothed > thresh

    # Clean up - use max_size for newer scikit-image
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=100)

    # Distance transform
    distance = ndimage.distance_transform_edt(binary)

    # Find local maxima as markers
    from skimage.feature import peak_local_max
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)

    # Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)

    # Remove small objects
    labels = morphology.remove_small_objects(labels, min_size=min_size)

    # Relabel consecutively
    labels = measure.label(labels > 0)

    return labels


def segment_cells_local_maxima(
    nuclear_image: np.ndarray,
    sigma: float = 1.0,
    min_distance: int = 2,
    threshold_rel: float = 0.05,
    expansion: str = 'voronoi',
) -> np.ndarray:
    """Segmentation for densely packed cells using local maxima detection.

    Better for lymphoma and other densely packed tissues where cells touch.

    Args:
        nuclear_image: 2D normalized nuclear image (0-1)
        sigma: Gaussian smoothing sigma (smaller = more sensitive to small cells)
        min_distance: minimum distance between cell centers in pixels
        threshold_rel: relative threshold for peak detection (0-1)
        expansion: 'voronoi' (expand to neighbors) or 'nuclear' (nuclear mask only)

    Returns:
        2D label mask
    """
    from scipy import ndimage
    from skimage import filters, segmentation
    from skimage.feature import peak_local_max

    # Smooth to reduce noise while preserving cell centers
    smoothed = ndimage.gaussian_filter(nuclear_image, sigma=sigma)

    # Find local maxima (cell centers)
    coords = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )

    # Create markers at cell centers
    markers = np.zeros(nuclear_image.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # Expand markers to full cells
    if expansion == 'voronoi':
        # Voronoi expansion - each pixel goes to nearest cell center
        # Use gradient as the landscape for watershed
        gradient = filters.sobel(smoothed)
        labels = segmentation.watershed(gradient, markers)
    else:
        # Nuclear mask only - use thresholding
        thresh = filters.threshold_otsu(smoothed) * 0.5
        mask = smoothed > thresh
        gradient = filters.sobel(smoothed)
        labels = segmentation.watershed(gradient, markers, mask=mask)

    return labels


def prepare_membrane_image(
    image: np.ndarray,
    markers: list[str],
    membrane_markers: list[str] = ['CD45RO', 'CD3', 'CD20'],
    percentile_norm: float = 99.5,
) -> np.ndarray:
    """Prepare membrane image by combining membrane channels.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        membrane_markers: markers to combine for membrane signal
        percentile_norm: percentile for normalization

    Returns:
        2D normalized membrane image
    """
    membrane_channels = []

    for marker in membrane_markers:
        if marker in markers:
            idx = markers.index(marker)
            channel = image[:, :, idx].astype(np.float32)
            # Normalize each channel individually before combining
            vmin = np.percentile(channel, 100 - percentile_norm)
            vmax = np.percentile(channel, percentile_norm)
            if vmax > vmin:
                channel = (channel - vmin) / (vmax - vmin)
            channel = np.clip(channel, 0, 1)
            membrane_channels.append(channel)

    if not membrane_channels:
        return None

    # Combine channels (max projection to capture any membrane signal)
    membrane = np.max(membrane_channels, axis=0)

    return membrane


def segment_cells_membrane(
    nuclear_image: np.ndarray,
    membrane_image: np.ndarray,
    sigma: float = 1.0,
    min_distance: int = 2,
    threshold_rel: float = 0.05,
    membrane_weight: float = 0.7,
) -> np.ndarray:
    """Segmentation using nuclear detection + membrane-guided boundaries.

    Uses nuclear signal to find cell centers, membrane signal to define boundaries.

    Args:
        nuclear_image: 2D normalized nuclear image (0-1)
        membrane_image: 2D normalized membrane image (0-1)
        sigma: Gaussian smoothing sigma
        min_distance: minimum distance between cell centers
        threshold_rel: relative threshold for peak detection
        membrane_weight: weight for membrane vs nuclear in gradient (0-1)

    Returns:
        2D label mask
    """
    from scipy import ndimage
    from skimage import filters, segmentation
    from skimage.feature import peak_local_max

    # Smooth nuclear image for cell center detection
    nuclear_smooth = ndimage.gaussian_filter(nuclear_image, sigma=sigma)

    # Find cell centers from nuclear signal
    coords = peak_local_max(
        nuclear_smooth,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )

    # Create markers at cell centers
    markers = np.zeros(nuclear_image.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # Create combined gradient for watershed
    # Membrane gradient defines boundaries, nuclear gradient helps where membrane is weak
    membrane_smooth = ndimage.gaussian_filter(membrane_image, sigma=0.5)
    membrane_gradient = filters.sobel(membrane_smooth)
    nuclear_gradient = filters.sobel(nuclear_smooth)

    # Combine gradients (higher membrane weight = boundaries follow membrane more)
    combined_gradient = (
        membrane_weight * membrane_gradient +
        (1 - membrane_weight) * nuclear_gradient
    )

    # Watershed on combined gradient
    labels = segmentation.watershed(combined_gradient, markers)

    return labels


def segment_cells_cellpose(
    nuclear_image: np.ndarray,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    model_type: str = 'nuclei',
    gpu: bool = False,
) -> np.ndarray:
    """Segment cells using Cellpose.

    Args:
        nuclear_image: 2D normalized nuclear image
        diameter: expected cell diameter in pixels (None for auto-detect)
        flow_threshold: flow error threshold (higher = more cells, less accurate)
        cellprob_threshold: cell probability threshold (lower = more cells)
        model_type: 'nuclei' or 'cyto' (ignored in v4.0+, nuclei model used by default)
        gpu: use GPU if available

    Returns:
        2D label mask where each cell has unique integer ID
    """
    from cellpose import models

    # Initialize model (Cellpose 4.0+ uses CellposeModel)
    # In v4.0+, nuclei model is the default
    model = models.CellposeModel(gpu=gpu)

    # Run segmentation
    # Cellpose expects image in range 0-255 or normalized
    img = (nuclear_image * 255).astype(np.uint8)

    # v4.0+ simplified API
    masks, flows, styles = model.eval(
        img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    return masks


def segment_cells_hybrid(
    nuclear_image: np.ndarray,
    sigma: float = 1.0,
    min_distance: int = 2,
    flow_threshold: float = 0.8,
    cellprob_threshold: float = 0.0,
    gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid segmentation: Cellpose boundaries filtered by nuclear detection.

    This approach:
    1. Detects nuclei from DNA signal (local maxima) - gives reliable cell count
    2. Runs Cellpose for cell boundaries
    3. Keeps only Cellpose cells that contain ≥1 nucleus

    Best for densely packed tissue like lymphoma where Cellpose may over-segment.

    Args:
        nuclear_image: 2D normalized nuclear image (0-1)
        sigma: Gaussian smoothing for nuclear detection
        min_distance: minimum distance between nuclei
        flow_threshold: Cellpose flow threshold (0.8 recommended for dense tissue)
        cellprob_threshold: Cellpose cell probability threshold
        gpu: use GPU for Cellpose

    Returns:
        tuple of (filtered_masks, nuclei_coords)
    """
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from cellpose import models

    # Step 1: Detect nuclei from DNA signal
    smoothed = ndimage.gaussian_filter(nuclear_image, sigma=sigma)
    nuclei_coords = peak_local_max(smoothed, min_distance=min_distance, threshold_rel=0.05)

    # Step 2: Run Cellpose
    model = models.CellposeModel(gpu=gpu)
    img = (nuclear_image * 255).astype(np.uint8)
    masks_cp, _, _ = model.eval(
        img,
        diameter=None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    # Step 3: Find which cells contain nuclei
    cells_with_nuclei = set()
    for y, x in nuclei_coords:
        cell_id = masks_cp[y, x]
        if cell_id > 0:
            cells_with_nuclei.add(cell_id)

    # Step 4: Filter masks - keep only cells with nuclei
    filtered_masks = np.zeros_like(masks_cp)
    new_id = 1
    for old_id in sorted(cells_with_nuclei):
        filtered_masks[masks_cp == old_id] = new_id
        new_id += 1

    return filtered_masks, nuclei_coords


def extract_cell_features(
    image: np.ndarray,
    markers: list[str],
    masks: np.ndarray,
    features: list[str] = ['mean', 'sum'],
) -> dict:
    """Extract marker intensities for each segmented cell.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        masks: 2D label mask from segmentation
        features: which features to extract ('mean', 'sum', 'median', 'max')

    Returns:
        dict with 'cell_ids', 'centroids', and feature matrices
    """
    from scipy import ndimage

    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]  # Remove background (0)
    n_cells = len(cell_ids)
    n_markers = len(markers)

    # Initialize feature arrays
    result = {
        'cell_ids': cell_ids,
        'markers': markers,
        'n_cells': n_cells,
    }

    # Calculate centroids
    centroids = ndimage.center_of_mass(masks > 0, masks, cell_ids)
    result['centroids'] = np.array(centroids)

    # Calculate cell areas
    areas = ndimage.sum(np.ones_like(masks), masks, cell_ids)
    result['area'] = np.array(areas)

    # Extract intensity features for each marker
    for feat in features:
        feat_matrix = np.zeros((n_cells, n_markers), dtype=np.float32)

        for j, marker in enumerate(markers):
            channel = image[:, :, j]

            if feat == 'mean':
                values = ndimage.mean(channel, masks, cell_ids)
            elif feat == 'sum':
                values = ndimage.sum(channel, masks, cell_ids)
            elif feat == 'median':
                values = ndimage.median(channel, masks, cell_ids)
            elif feat == 'max':
                values = ndimage.maximum(channel, masks, cell_ids)
            else:
                raise ValueError(f"Unknown feature: {feat}")

            feat_matrix[:, j] = values

        result[feat] = feat_matrix

    return result


def features_to_anndata(
    features: dict,
    sample_id: str,
    roi_metadata: Optional[dict] = None,
):
    """Convert extracted features to AnnData format.

    Args:
        features: output from extract_cell_features
        sample_id: sample identifier
        roi_metadata: optional ROI metadata

    Returns:
        AnnData object
    """
    import anndata as ad
    import pandas as pd

    # Use mean intensities as main matrix
    X = features['mean']
    n_cells = features['n_cells']

    # Cell metadata
    obs_data = {
        'cell_id': features['cell_ids'],
        'sample_id': [sample_id] * n_cells,
        'centroid_y': features['centroids'][:, 0],
        'centroid_x': features['centroids'][:, 1],
        'area': features['area'],
    }
    obs = pd.DataFrame(obs_data)
    obs.index = [f"{sample_id}_{i}" for i in features['cell_ids']]

    # Marker metadata
    var = pd.DataFrame({'marker': features['markers']})
    var.index = features['markers']

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Store sum intensities in layers
    if 'sum' in features:
        adata.layers['sum'] = features['sum']

    # Store spatial coordinates
    adata.obsm['spatial'] = features['centroids'][:, ::-1]  # (x, y) format

    # Store ROI metadata
    if roi_metadata:
        adata.uns['roi_metadata'] = roi_metadata

    return adata


def segment_roi(
    image: np.ndarray,
    markers: list[str],
    sample_id: str,
    method: str = 'local_maxima',
    diameter: Optional[float] = None,
    nuclear_markers: list[str] = ['DNA1', 'DNA2'],
    membrane_markers: list[str] = ['CD45RO', 'CD3', 'CD20'],
    membrane_weight: float = 0.7,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_distance: int = 2,
    sigma: float = 1.0,
    expansion: str = 'voronoi',
    gpu: bool = False,
    roi_metadata: Optional[dict] = None,
):
    """Full segmentation pipeline for a single ROI.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        sample_id: sample identifier
        method: 'hybrid' (recommended), 'local_maxima', 'membrane', 'watershed', or 'cellpose'
        diameter: expected cell diameter for Cellpose (None for auto)
        nuclear_markers: markers to use for nuclear signal
        membrane_markers: markers for membrane signal (for 'membrane' method)
        membrane_weight: weight for membrane vs nuclear gradient (0-1)
        flow_threshold: Cellpose flow threshold
        cellprob_threshold: Cellpose cell probability threshold
        min_distance: minimum distance between cell centers (for local_maxima/watershed)
        sigma: Gaussian smoothing sigma (for local_maxima)
        expansion: 'voronoi' or 'nuclear' (for local_maxima)
        gpu: use GPU for Cellpose
        roi_metadata: optional metadata

    Returns:
        tuple of (masks, adata)
    """
    # Prepare nuclear image
    nuclear = prepare_nuclear_image(image, markers, nuclear_markers)

    # Segment
    if method == 'membrane':
        # Use membrane markers for boundary definition
        membrane = prepare_membrane_image(image, markers, membrane_markers)
        if membrane is None:
            print(f"Warning: No membrane markers found, falling back to local_maxima")
            masks = segment_cells_local_maxima(
                nuclear, sigma=sigma, min_distance=min_distance, expansion=expansion
            )
        else:
            masks = segment_cells_membrane(
                nuclear, membrane,
                sigma=sigma,
                min_distance=min_distance,
                membrane_weight=membrane_weight,
            )
    elif method == 'local_maxima':
        masks = segment_cells_local_maxima(
            nuclear,
            sigma=sigma,
            min_distance=min_distance,
            expansion=expansion,
        )
    elif method == 'watershed':
        masks = segment_cells_watershed(
            nuclear,
            min_distance=min_distance,
        )
    elif method == 'cellpose':
        masks = segment_cells_cellpose(
            nuclear,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            gpu=gpu,
        )
    elif method == 'hybrid':
        # Cellpose boundaries filtered by nuclear detection
        masks, nuclei_coords = segment_cells_hybrid(
            nuclear,
            sigma=sigma,
            min_distance=min_distance,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            gpu=gpu,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hybrid', 'membrane', 'local_maxima', 'watershed', or 'cellpose'")

    # Extract features
    features = extract_cell_features(image, markers, masks)

    # Convert to AnnData
    adata = features_to_anndata(features, sample_id, roi_metadata)

    return masks, adata


def visualize_segmentation(
    image: np.ndarray,
    markers: list[str],
    masks: np.ndarray,
    nuclear_markers: list[str] = ['DNA1', 'DNA2'],
    overlay_alpha: float = 0.3,
    figsize: tuple = (16, 8),
    show_boundaries: bool = True,
):
    """Visualize segmentation results.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        masks: segmentation masks
        nuclear_markers: markers used for nuclear signal
        overlay_alpha: transparency for mask overlay
        figsize: figure size
        show_boundaries: show cell boundaries instead of filled masks
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage import segmentation

    # Prepare nuclear image
    nuclear = prepare_nuclear_image(image, markers, nuclear_markers)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Nuclear image
    axes[0].imshow(nuclear, cmap='gray')
    axes[0].set_title(f'Nuclear ({"+".join(nuclear_markers)})')
    axes[0].axis('off')

    # 2. Segmentation masks
    # Create random colormap for cells
    n_cells = masks.max()
    np.random.seed(42)
    colors = np.random.rand(n_cells + 1, 3)
    colors[0] = [0, 0, 0]  # Background black
    cmap = ListedColormap(colors)

    if show_boundaries:
        boundaries = segmentation.find_boundaries(masks, mode='outer')
        axes[1].imshow(nuclear, cmap='gray')
        axes[1].imshow(boundaries, cmap='Reds', alpha=0.7)
    else:
        axes[1].imshow(masks, cmap=cmap, interpolation='nearest')

    axes[1].set_title(f'Segmentation ({n_cells} cells)')
    axes[1].axis('off')

    # 3. Overlay
    axes[2].imshow(nuclear, cmap='gray')
    mask_overlay = np.ma.masked_where(masks == 0, masks)
    axes[2].imshow(mask_overlay, cmap=cmap, alpha=overlay_alpha, interpolation='nearest')
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def parameter_sweep(
    image: np.ndarray,
    markers: list[str],
    diameters: list[float] = [20, 30, 40, 50],
    nuclear_markers: list[str] = ['DNA1', 'DNA2'],
    figsize: tuple = (16, 12),
):
    """Run segmentation with different diameter parameters to help choose.

    Args:
        image: 3D array (H, W, C)
        markers: list of marker names
        diameters: list of diameters to try
        nuclear_markers: markers for nuclear signal
        figsize: figure size
    """
    import matplotlib.pyplot as plt
    from skimage import segmentation

    nuclear = prepare_nuclear_image(image, markers, nuclear_markers)

    n = len(diameters)
    fig, axes = plt.subplots(2, n, figsize=figsize)

    for i, diam in enumerate(diameters):
        print(f"Testing diameter={diam}...")
        masks = segment_cells_cellpose(nuclear, diameter=diam)
        n_cells = masks.max()

        # Top row: boundaries on nuclear
        boundaries = segmentation.find_boundaries(masks, mode='outer')
        axes[0, i].imshow(nuclear, cmap='gray')
        axes[0, i].imshow(boundaries, cmap='Reds', alpha=0.7)
        axes[0, i].set_title(f'd={diam}, n={n_cells}')
        axes[0, i].axis('off')

        # Bottom row: zoomed region
        h, w = nuclear.shape
        y1, y2 = h // 3, 2 * h // 3
        x1, x2 = w // 3, 2 * w // 3

        axes[1, i].imshow(nuclear[y1:y2, x1:x2], cmap='gray')
        axes[1, i].imshow(boundaries[y1:y2, x1:x2], cmap='Reds', alpha=0.7)
        axes[1, i].set_title(f'Zoomed center')
        axes[1, i].axis('off')

    plt.suptitle('Diameter Parameter Sweep', fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Quick test
    import sys
    sys.path.insert(0, '.')
    from src.data_loader import load_roi_txt, list_rois, extract_sample_id

    data_dir = Path('data/raw/TMA_B1_T')
    files = list_rois(data_dir)

    print(f"Loading {files[0].name}...")
    image, markers, metadata = load_roi_txt(files[0])
    sample_id = extract_sample_id(files[0].name)

    print(f"Running segmentation...")
    masks, adata = segment_roi(image, markers, sample_id, diameter=30)

    print(f"Segmented {adata.n_obs} cells")
    print(f"AnnData: {adata}")

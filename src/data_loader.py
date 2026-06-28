"""
Data loader for Hyperion IMC pixel-level TXT exports.

Converts tab-separated pixel intensity files to numpy arrays and TIFF images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import re
from tqdm import tqdm


def parse_marker_name(column: str) -> str:
    """Extract clean marker name from column header.

    Example: 'CD3(Er170Di)' -> 'CD3'
    """
    match = re.match(r'^([^(]+)', column)
    return match.group(1) if match else column


def load_roi_txt(filepath: Path | str) -> tuple[np.ndarray, list[str], dict]:
    """Load a single ROI TXT file.

    Args:
        filepath: Path to the TXT file

    Returns:
        tuple of:
            - image: 3D numpy array (height, width, channels)
            - markers: list of marker names
            - metadata: dict with ROI info
    """
    filepath = Path(filepath)

    # Load as dataframe
    df = pd.read_csv(filepath, sep='\t')

    # Extract coordinates
    x = df['X'].values
    y = df['Y'].values

    # Get image dimensions
    width = x.max() + 1
    height = y.max() + 1

    # Identify marker columns (skip metadata columns)
    skip_cols = {'Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z'}
    marker_cols = [c for c in df.columns if c not in skip_cols]
    markers = [parse_marker_name(c) for c in marker_cols]

    # Build image array
    n_channels = len(marker_cols)
    image = np.zeros((height, width, n_channels), dtype=np.float32)

    for i, col in enumerate(marker_cols):
        image[y, x, i] = df[col].values

    # Extract metadata from filename
    metadata = {
        'filename': filepath.name,
        'width': width,
        'height': height,
        'n_pixels': len(df),
        'n_channels': n_channels,
    }

    return image, markers, metadata


def load_roi_as_dict(filepath: Path | str) -> dict:
    """Load ROI and return as dictionary with all components."""
    image, markers, metadata = load_roi_txt(filepath)
    return {
        'image': image,
        'markers': markers,
        'metadata': metadata,
    }


def get_channel_image(image: np.ndarray, markers: list[str], marker: str) -> np.ndarray:
    """Extract a single channel from the image by marker name."""
    if marker not in markers:
        available = ', '.join(markers[:10]) + '...'
        raise ValueError(f"Marker '{marker}' not found. Available: {available}")
    idx = markers.index(marker)
    return image[:, :, idx]


def list_rois(data_dir: Path | str, panel: Optional[str] = None) -> list[Path]:
    """List all ROI TXT files in a directory.

    Args:
        data_dir: Directory containing TXT files
        panel: Optional filter - 'T' for T-cell panel, 'S' for Stromal panel

    Returns:
        List of file paths
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('*.txt'))

    if panel:
        panel_lower = panel.lower()
        if panel_lower == 't':
            files = [f for f in files if 'tcellpanel' in f.name.lower()]
        elif panel_lower == 's':
            files = [f for f in files if 'stromalpanel' in f.name.lower()]

    return files


def extract_sample_id(filename: str) -> str:
    """Extract sample ID from filename.

    Examples:
        '20210518_CT14_09_B1_Stromalpanel_1_FL1_L_3.txt' -> 'FL1'
        '20220118_CT14_09_B1_Tcellpanel_1_FL01_L_4.txt' -> 'FL01'
    """
    # Look for FL followed by numbers
    match = re.search(r'(FL\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: look for ROI pattern
    match = re.search(r'ROI_(\d+)', filename)
    if match:
        return f"ROI_{match.group(1)}"

    return filename


class IMCDataset:
    """Dataset class for managing multiple ROIs."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.files = list_rois(self.data_dir)
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        filepath = self.files[idx]
        if filepath not in self._cache:
            self._cache[filepath] = load_roi_as_dict(filepath)
        return self._cache[filepath]

    def get_by_sample(self, sample_id: str) -> Optional[dict]:
        """Get ROI by sample ID (e.g., 'FL1')."""
        for f in self.files:
            if extract_sample_id(f.name).upper() == sample_id.upper():
                return load_roi_as_dict(f)
        return None

    def list_samples(self) -> list[str]:
        """List all sample IDs."""
        return [extract_sample_id(f.name) for f in self.files]

    def clear_cache(self):
        """Clear the loaded data cache."""
        self._cache.clear()


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Loading {filepath}...")
        image, markers, metadata = load_roi_txt(filepath)
        print(f"Image shape: {image.shape}")
        print(f"Markers ({len(markers)}): {markers[:10]}...")
        print(f"Metadata: {metadata}")

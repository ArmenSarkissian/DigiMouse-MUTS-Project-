"""
digimouse_tools.py

Python port of the MATLAB Digimouse workflow:
  - loading ANALYZE 7.5 (.hdr/.img) volumes (atlas, CT, PET, cryosection)
  - identifying anatomical planes (coronal / sagittal / axial)
  - static 3-view plotting with a fixed, consistent label colormap
  - interactive slice scrubbing (matplotlib Slider widget)
  - overlaying atlas labels on top of a real CT/PET/cryosection volume

Requires: nibabel, numpy, matplotlib
    pip install nibabel numpy matplotlib

Axis convention for the Digimouse atlas (confirmed empirically from the
.hdr header, matrix size 380 x 992 x 208):
    axis 0 (length 380)  -> left-right     (sagittal axis)
    axis 1 (length 992)  -> nose-to-tail   (coronal / anterior-posterior axis)
    axis 2 (length 208)  -> dorsal-ventral (axial / top-to-bottom axis)

This matches standard rodent neuroimaging convention:
    coronal  = fix the AP axis   (axis 1) -> circular/oval cross-section
    sagittal = fix the L-R axis  (axis 0) -> side profile
    axial    = fix the D-V axis  (axis 2) -> top-down view
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider


ORGAN_LABELS = [
    "background", "skin", "skeleton", "eye", "medulla", "cerebellum",
    "olfactory bulbs", "external cerebrum", "striatum", "heart",
    "rest of brain", "masseter muscles", "lachrymal glands", "bladder",
    "testis", "stomach", "spleen", "pancreas", "liver", "kidneys",
    "adrenal glands", "lungs",
]
N_LABELS = len(ORGAN_LABELS)  # 22 (0-21)

# Plane name -> which array axis is held fixed when slicing that plane
PLANE_AXIS = {"coronal": 1, "sagittal": 0, "axial": 2}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_analyze(hdr_path):
    """
    Load an ANALYZE 7.5 (.hdr/.img pair) volume, equivalent to MATLAB's
        info = analyze75info('file.hdr');
        X    = analyze75read(info);

    Returns a 3D numpy array in its native dtype (e.g. uint8 for the atlas
    labels, float for CT/PET), with any trailing singleton 4th dimension
    squeezed out.
    """
    img = nib.load(hdr_path)
    vol = np.asanyarray(img.dataobj)
    vol = np.squeeze(vol)
    return vol


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------
def get_slice(vol, plane, idx):
    """Return a 2D slice of `vol` for the given anatomical plane and index."""
    if plane == "coronal":
        return vol[:, idx, :]
    elif plane == "sagittal":
        return vol[idx, :, :]
    elif plane == "axial":
        return vol[:, :, idx]
    else:
        raise ValueError("plane must be 'coronal', 'sagittal', or 'axial'")


def n_slices(vol, plane):
    return vol.shape[PLANE_AXIS[plane]]


# ---------------------------------------------------------------------------
# Static 3-view plot (fixed colormap/caxis, like the MATLAB caxis([0 21]) code)
# ---------------------------------------------------------------------------
def plot_three_views(vol, coronal_idx=None, sagittal_idx=None, axial_idx=None,
                      labels=ORGAN_LABELS, save_path=None):
    """
    Plot coronal / sagittal / axial slices of a labeled atlas volume side by
    side, using one fixed discrete colormap so the same label is always the
    same color across all three views (equivalent to the MATLAB
    `colormap(jet(22)); caxis([0 21])` pattern).
    """
    if coronal_idx is None:
        coronal_idx = vol.shape[1] // 2
    if sagittal_idx is None:
        sagittal_idx = vol.shape[0] // 2
    if axial_idx is None:
        axial_idx = vol.shape[2] // 2

    n = len(labels)
    cmap = plt.get_cmap("jet", n)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    views = [
        ("coronal", coronal_idx, vol.shape[1]),
        ("sagittal", sagittal_idx, vol.shape[0]),
        ("axial", axial_idx, vol.shape[2]),
    ]
    im = None
    for ax, (plane, idx, total) in zip(axes, views):
        im = ax.imshow(get_slice(vol, plane, idx).T, cmap=cmap, norm=norm, origin="lower")
        ax.set_title(f"{plane.capitalize()} (slice {idx} of {total})")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, ticks=range(n), fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels([f"{i} {name}" for i, name in enumerate(labels)], fontsize=7)

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, axes


# ---------------------------------------------------------------------------
# Interactive slice viewer (equivalent to the MATLAB uicontrol slider code)
# ---------------------------------------------------------------------------
def slice_viewer(vol, plane, labels=ORGAN_LABELS):
    """
    Open an interactive matplotlib window with a slider to scrub through
    every slice of `vol` along the given anatomical plane. Equivalent to
    the MATLAB sliceSlider() function from the earlier MATLAB workflow.

    plane: 'coronal', 'sagittal', or 'axial'
    """
    n = len(labels)
    cmap = plt.get_cmap("jet", n)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n + 0.5, 1), cmap.N)

    total = n_slices(vol, plane)
    start_idx = total // 2

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.18)

    img_artist = ax.imshow(get_slice(vol, plane, start_idx).T, cmap=cmap, norm=norm, origin="lower")
    ax.set_title(f"{plane.capitalize()} \u2014 slice {start_idx} of {total}")
    ax.axis("off")

    cbar = fig.colorbar(img_artist, ax=ax, ticks=range(n), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f"{i} {name}" for i, name in enumerate(labels)], fontsize=6)

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, "Slice", 0, total - 1, valinit=start_idx, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_artist.set_data(get_slice(vol, plane, idx).T)
        ax.set_title(f"{plane.capitalize()} \u2014 slice {idx} of {total}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return fig, ax, slider  # keep references alive (esp. `slider`) if embedding


# ---------------------------------------------------------------------------
# Atlas-on-CT (or PET / cryosection) overlay
# ---------------------------------------------------------------------------
def overlay_atlas_on_background(background_vol, atlas_vol, plane, idx,
                                 labels=ORGAN_LABELS, alpha=0.45, save_path=None):
    """
    Blend the atlas's organ labels as a colored, semi-transparent overlay on
    top of a coregistered grayscale volume (CT, PET, or cryosection),
    equivalent to the MATLAB RGB-overlay code (ind2rgb + AlphaData).

    background_vol and atlas_vol must already be coregistered to the same
    grid (true for the Digimouse atlas + ct_data/pet_data/cryo_data).
    """
    bg_slice = get_slice(background_vol, plane, idx).astype(float).T
    atlas_slice = get_slice(atlas_vol, plane, idx).T

    # Normalize background to [0, 1] grayscale
    bg_norm = (bg_slice - bg_slice.min()) / (np.ptp(bg_slice) + 1e-12)
    bg_rgb = np.stack([bg_norm] * 3, axis=-1)

    # Map atlas labels to RGB via the same fixed discrete colormap
    n = len(labels)
    cmap = plt.get_cmap("jet", n)
    atlas_rgb = cmap(atlas_slice.astype(int) / (n - 1))[..., :3]

    # Alpha mask: fully transparent where label == 0 (background)
    mask = (atlas_slice > 0).astype(float) * alpha

    blended = bg_rgb * (1 - mask[..., None]) + atlas_rgb * mask[..., None]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(blended, origin="lower")
    ax.set_title(f"Atlas labels overlaid on background ({plane}, slice {idx})")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hdr_path = r"C:\Users\armen\OneDrive\Documents\Digimouse_Data\atlas\atlas\atlas_380x992x208.hdr"
    vol = load_analyze(hdr_path)
    print("Loaded volume shape:", vol.shape)
    print("Unique labels:", np.unique(vol))

    plot_three_views(vol, coronal_idx=500, sagittal_idx=190, axial_idx=100)
    plt.show()  # opens the plot in a window

    # Uncomment to open the interactive slider viewer instead:
    # slice_viewer(vol, "coronal")

"""
digimouse_3d.py

3D rendering of the labeled Digimouse atlas with per-organ show/hide.

Requires:  pip install numpy nibabel scipy pyvista
(scikit-image is NOT needed; PyVista/VTK does the surface extraction.)

Axis convention (same as digimouse_tools.py):
    axis 0 = left-right (380), axis 1 = nose-to-tail (992), axis 2 = dorsal-ventral (208)
Voxels are 0.1 mm, so all 3D coordinates below are in millimetres.
"""

import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ORGAN_LABELS = [
    "background", "skin", "skeleton", "eye", "medulla", "cerebellum",
    "olfactory bulbs", "external cerebrum", "striatum", "heart",
    "rest of brain", "masseter muscles", "lachrymal glands", "bladder",
    "testis", "stomach", "spleen", "pancreas", "liver", "kidneys",
    "adrenal glands", "lungs",
]
VOXEL_MM = 0.1
SKIN = 1

# Same 22-color jet map as the 2D viewers, so an organ keeps its color in 2D and 3D.
_CMAP = plt.get_cmap("jet", len(ORGAN_LABELS))


def label_color(label):
    if label == SKIN:
        return (0.93, 0.80, 0.70)  # neutral beige instead of dark blue; skin is drawn faint anyway
    return _CMAP(label)[:3]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_analyze(hdr_path):
    img = nib.load(hdr_path)
    return np.squeeze(np.asanyarray(img.dataobj))


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------
def build_organ_mesh(vol, label, step=2, smooth_sigma=1.0, pad=3):
    """
    Build one smooth surface mesh (pv.PolyData, in mm) for a single label.

    step          : keep every `step`-th voxel (2 -> 0.2 mm resolution, ~8x faster
                    and lighter than full resolution). Use 1 for full detail.
    smooth_sigma  : Gaussian smoothing (in downsampled voxels) applied to the binary
                    mask before extracting the surface; removes the stair-step look.
    Returns None if the label is absent from the volume.
    """
    mask = vol == label
    if not mask.any():
        return None

    # Crop to the label's bounding box (plus padding) to save memory/time
    idx = np.where(mask)
    lo = [max(int(i.min()) - pad * step, 0) for i in idx]
    hi = [min(int(i.max()) + pad * step + 1, s) for i, s in zip(idx, vol.shape)]
    crop = mask[lo[0]:hi[0]:step, lo[1]:hi[1]:step, lo[2]:hi[2]:step].astype(np.float32)

    # Pad with zeros so the surface closes at the crop boundary
    crop = np.pad(crop, 1)
    if smooth_sigma > 0:
        crop = gaussian_filter(crop, smooth_sigma)

    spacing = VOXEL_MM * step
    grid = pv.ImageData(dimensions=crop.shape, spacing=(spacing,) * 3)
    grid.origin = tuple((l - step) * VOXEL_MM for l in lo)  # -1 padded voxel
    grid.point_data["m"] = crop.flatten(order="F")

    mesh = grid.contour([0.5], scalars="m", method="flying_edges")
    if mesh.n_points == 0:
        return None
    return mesh.compute_normals(auto_orient_normals=True)


def build_all_meshes(vol, labels=None, **kwargs):
    """Dict {label: PolyData} for every label present (skips background)."""
    if labels is None:
        labels = [int(l) for l in np.unique(vol) if l != 0]
    meshes = {}
    for lab in labels:
        m = build_organ_mesh(vol, lab, **kwargs)
        if m is not None:
            meshes[lab] = m
            print(f"  built {lab:2d} {ORGAN_LABELS[lab]:<18} {m.n_points:>8} points")
    return meshes


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
class OrganScene:
    """Holds a PyVista plotter with one actor per organ and visibility control."""

    def __init__(self, meshes, off_screen=False, skin_opacity=0.15, window_size=(1300, 800)):
        self.meshes = meshes
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        self.plotter.set_background("white")
        self.actors = {}
        self.checkboxes = {}

        for lab, mesh in meshes.items():
            opacity = skin_opacity if lab == SKIN else 1.0
            self.actors[lab] = self.plotter.add_mesh(
                mesh, color=label_color(lab), opacity=opacity, smooth_shading=True,
                name=ORGAN_LABELS[lab],
            )

    # --- visibility -----------------------------------------------------
    def set_visible(self, label, visible):
        """Show/hide one organ and keep its checkbox in sync."""
        if label in self.actors:
            self.actors[label].SetVisibility(bool(visible))
        cb = self.checkboxes.get(label)
        if cb is not None:
            cb.GetRepresentation().SetState(int(bool(visible)))
        self.plotter.render()

    def show_only(self, labels):
        """Show just these labels, hide the rest."""
        keep = set(labels)
        for lab in self.actors:
            self.set_visible(lab, lab in keep)

    def show_all(self):
        self.show_only(self.actors.keys())

    def hide_all(self):
        self.show_only([])

    # --- interactive controls ------------------------------------------
    def add_controls(self):
        p = self.plotter
        labs = sorted(self.actors)
        x, y0, dy = 10, 760, 27
        for i, lab in enumerate(labs):
            y = y0 - i * dy
            self.checkboxes[lab] = p.add_checkbox_button_widget(
                lambda state, l=lab: self.set_visible(l, state),
                value=True, position=(x, y), size=20, border_size=2,
                color_on=label_color(lab), color_off="lightgray", background_color="white",
            )
            p.add_text(ORGAN_LABELS[lab], position=(x + 28, y + 2), font_size=8,
                       color="black", name=f"lbl_{lab}")

        # Extra buttons (key presses)
        p.add_key_event("a", self.show_all)
        p.add_key_event("h", self.hide_all)
        p.add_text("Keys:  a = show all   h = hide all", position="upper_right",
                   font_size=8, color="gray", name="help")

        if SKIN in self.actors:
            p.add_slider_widget(
                lambda v: setattr(self.actors[SKIN].prop, "opacity", float(v)),
                [0.0, 1.0], value=0.15, title="Skin opacity",
                pointa=(0.60, 0.08), pointb=(0.95, 0.08), style="modern",
            )

    def show(self):
        self.add_controls()
        self.plotter.view_isometric()
        self.plotter.show()

    def screenshot(self, path, camera="isometric"):
        {"isometric": self.plotter.view_isometric,
         "side": self.plotter.view_yz,
         "top": self.plotter.view_xy}[camera]()
        self.plotter.show(screenshot=path, auto_close=False)


# ---------------------------------------------------------------------------
# Run it (edit the path below, then press Run in VS Code)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    HDR_PATH = r"C:\Users\armen\Documents\_temp_matlab_R2024b_Windows\atlas\atlas\atlas_380x992x208.hdr"

    vol = load_analyze(HDR_PATH)
    print("Loaded", vol.shape, "- building meshes (takes a little while)...")
    meshes = build_all_meshes(vol, step=2)

    scene = OrganScene(meshes)
    scene.show()
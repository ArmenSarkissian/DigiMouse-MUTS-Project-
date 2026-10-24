"""
digimouse_3d.py  (interactive version)

3D viewer for the labeled Digimouse atlas.

Features
  * Click an organ to select it (Shift+click to add more). Selected organs are
    highlighted (bounding box + others ghosted).
  * "Isolate selected": show ONLY the selected organ(s) and zoom to them.
  * Real-world scale: 10 mm scale bar, optional mm ruler grid, and size / volume
    read-outs for the whole mouse and for each selected organ.
  * Slice planes inside the 3D scene (coronal / sagittal / axial) with sliders,
    exact-index control, and an optional "cut-away" so the slice is not hidden
    behind the organs in front of it.

Requires:  pip install numpy nibabel scipy vtk pyvista matplotlib
(Use Python 3.13 or older: VTK has no wheels for 3.15 yet.)

Axis convention (same as digimouse_tools.py):
    axis 0 = left-right (380), axis 1 = nose-to-tail (992), axis 2 = dorsal-ventral (208)
Voxels are 0.1 mm, so every 3D coordinate is in millimetres.
"""

import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkRenderingCore import vtkPropPicker

ORGAN_LABELS = [
    "background", "skin", "skeleton", "eye", "medulla", "cerebellum",
    "olfactory bulbs", "external cerebrum", "striatum", "heart",
    "rest of brain", "masseter muscles", "lachrymal glands", "bladder",
    "testis", "stomach", "spleen", "pancreas", "liver", "kidneys",
    "adrenal glands", "lungs",
]
N_LABELS = len(ORGAN_LABELS)
VOXEL_MM = 0.1
SKIN = 1

# plane name -> (array axis held fixed, 3D axis it corresponds to)
PLANES = {"coronal": 1, "sagittal": 0, "axial": 2}
PLANE_TITLES = {"coronal": "Coronal", "sagittal": "Sagittal", "axial": "Axial"}

_CMAP = plt.get_cmap("jet", N_LABELS)
_LUT = np.array(
    [(0.93, 0.80, 0.70) if i == SKIN else _CMAP(i)[:3] for i in range(N_LABELS)]
)  # same colors as the 2D viewers (skin is beige instead of dark blue)
_LUT_U8 = (_LUT * 255).astype(np.uint8)


def label_color(label):
    return tuple(_LUT[label])


# ---------------------------------------------------------------------------
# Loading + mesh building
# ---------------------------------------------------------------------------
def load_analyze(hdr_path):
    img = nib.load(hdr_path)
    return np.squeeze(np.asanyarray(img.dataobj))


def build_organ_mesh(vol, label, step=2, smooth_sigma=1.0, pad=3):
    """One smooth surface (pv.PolyData, in mm) for a single label; None if absent."""
    mask = vol == label
    if not mask.any():
        return None
    idx = np.where(mask)
    lo = [max(int(i.min()) - pad * step, 0) for i in idx]
    hi = [min(int(i.max()) + pad * step + 1, s) for i, s in zip(idx, vol.shape)]
    crop = mask[lo[0]:hi[0]:step, lo[1]:hi[1]:step, lo[2]:hi[2]:step].astype(np.float32)
    crop = np.pad(crop, 1)
    if smooth_sigma > 0:
        crop = gaussian_filter(crop, smooth_sigma)
    spacing = VOXEL_MM * step
    grid = pv.ImageData(dimensions=crop.shape, spacing=(spacing,) * 3)
    grid.origin = tuple((l - step) * VOXEL_MM for l in lo)
    grid.point_data["m"] = crop.flatten(order="F")
    mesh = grid.contour([0.5], scalars="m", method="flying_edges")
    if mesh.n_points == 0:
        return None
    return mesh.compute_normals(auto_orient_normals=True)


def build_all_meshes(vol, labels=None, verbose=True, **kwargs):
    if labels is None:
        labels = [int(l) for l in np.unique(vol) if l != 0]
    meshes = {}
    for lab in labels:
        m = build_organ_mesh(vol, lab, **kwargs)
        if m is not None:
            meshes[lab] = m
            if verbose:
                print(f"  built {lab:2d} {ORGAN_LABELS[lab]:<18} {m.n_points:>8} points")
    return meshes


# ---------------------------------------------------------------------------
# The interactive scene
# ---------------------------------------------------------------------------
class OrganScene:
    GHOST_OPACITY = 0.10      # opacity of unselected organs while something is selected
    CLICK_TOLERANCE_PX = 4    # a press/release this close together counts as a click

    def __init__(self, vol, meshes, off_screen=False, window_size=(1400, 860),
                 skin_opacity=0.15, slice_step=1):
        self.vol = vol
        self.meshes = meshes
        self.window_size = window_size
        self.skin_opacity = skin_opacity
        self.slice_step = slice_step

        counts = np.bincount(vol.ravel(), minlength=N_LABELS)
        self.volume_mm3 = {l: counts[l] * VOXEL_MM ** 3 for l in meshes}   # 1 mm^3 = 1 microlitre

        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        self.plotter.set_background("white")

        # --- state ---
        self.visible = {l: True for l in meshes}
        self.selected = set()
        self.isolate = False
        self.active_plane = "coronal"
        self.slices = {
            p: dict(index=vol.shape[ax] // 2, show=False, cut=False, side=1)
            for p, ax in PLANES.items()
        }
        self.vis_boxes, self.sel_boxes, self.slice_widgets = {}, {}, {}
        self.slice_show_boxes, self.slice_cut_boxes = {}, {}
        self.isolate_box = self.ruler_box = None
        self._ruler_on = False
        self._ui_ready = False   # widget callbacks are ignored until add_controls() finishes

        # --- organ actors ---
        self.actors = {}
        for lab, mesh in meshes.items():
            actor = self.plotter.add_mesh(
                mesh, color=label_color(lab), smooth_shading=True,
                opacity=skin_opacity if lab == SKIN else 1.0, name=ORGAN_LABELS[lab],
            )
            # The faint skin shell must not swallow clicks meant for the organs inside it
            actor.SetPickable(lab != SKIN)
            self.actors[lab] = actor
        self._label_of_actor = {id(a): l for l, a in self.actors.items()}

        ref = meshes.get(SKIN) or next(iter(meshes.values()))
        self.body_bounds = ref.bounds
        self._add_scale_bar()
        self.plotter.add_axes(xlabel="L-R", ylabel="Nose-Tail", zlabel="D-V", line_width=3,
                              viewport=(0.0, 0.0, 0.13, 0.15))
        self._update_info()

    # ====================================================================
    # Organ visibility / selection / isolate
    # ====================================================================
    def set_visible(self, label, visible):
        """Show/hide one organ (independent of selection)."""
        if label in self.visible:
            self.visible[label] = bool(visible)
            self._sync_box(self.vis_boxes, label, visible)
            self._apply_state()

    def show_only(self, labels):
        keep = set(labels)
        for lab in self.visible:
            self.visible[lab] = lab in keep
            self._sync_box(self.vis_boxes, lab, lab in keep)
        self._apply_state()

    def show_all(self):
        self.show_only(self.visible.keys())

    def hide_all(self):
        self.show_only([])

    def select(self, labels, add=False):
        """Select organ(s). add=True keeps the current selection too."""
        labels = {l for l in labels if l in self.meshes}
        self.selected = (self.selected | labels) if add else labels
        self._after_selection_change()

    def toggle_select(self, label):
        if label in self.meshes:
            self.selected ^= {label}
            self._after_selection_change()

    def clear_selection(self):
        self.selected = set()
        self._after_selection_change()

    def set_isolate(self, on):
        self.isolate = bool(on)
        if self.isolate_box is not None:
            self.isolate_box.GetRepresentation().SetState(int(self.isolate))
        self._after_selection_change()

    def _after_selection_change(self):
        for lab in self.meshes:
            self._sync_box(self.sel_boxes, lab, lab in self.selected)
        self._apply_state()
        self._focus()

    def _apply_state(self):
        sel = self.selected
        for lab, actor in self.actors.items():
            show = self.visible[lab]
            if self.isolate and sel:
                show = show and lab in sel
            actor.SetVisibility(show)

            if lab == SKIN:
                opacity = self.skin_opacity
            elif sel and not self.isolate and lab not in sel:
                opacity = self.GHOST_OPACITY
            else:
                opacity = 1.0
            actor.prop.opacity = opacity
            actor.prop.ambient = 0.35 if lab in sel else 0.0   # selected organs look brighter

            # bounding box around each selected, visible organ
            name = f"selbox_{lab}"
            if lab in sel and show:
                self.plotter.add_mesh(self.meshes[lab].outline(), color="black", line_width=2,
                                      name=name, pickable=False, render=False)
            else:
                self.plotter.remove_actor(name, render=False)
        self._update_info()
        self.plotter.render()

    def _current_bounds(self):
        """Bounds of the selection when isolating, otherwise of the whole mouse."""
        if self.isolate and self.selected:
            bs = [self.meshes[l].bounds for l in self.selected]
            return (min(x[0] for x in bs), max(x[1] for x in bs),
                    min(x[2] for x in bs), max(x[3] for x in bs),
                    min(x[4] for x in bs), max(x[5] for x in bs))
        return self.body_bounds

    def _focus(self):
        """Zoom to the selection when isolating; otherwise frame the whole mouse."""
        b = self._current_bounds()
        self.plotter.reset_camera(bounds=b)
        self.plotter.camera.zoom(0.8)                 # leave a margin around the subject
        self._add_scale_bar(b)                        # scale bar re-sizes to what you are looking at
        if self._ruler_on:
            self._draw_ruler()
        self.plotter.render()

    def _update_info(self):
        bb = self.body_bounds
        lines = [f"Whole mouse: {bb[1]-bb[0]:.0f} x {bb[3]-bb[2]:.0f} x {bb[5]-bb[4]:.0f} mm "
                 f"(L-R x nose-tail x D-V)"]
        for lab in sorted(self.selected):
            b = self.meshes[lab].bounds
            lines.append(f"{ORGAN_LABELS[lab]}: {b[1]-b[0]:.1f} x {b[3]-b[2]:.1f} x {b[5]-b[4]:.1f} mm, "
                         f"{self.volume_mm3[lab]:,.0f} mm3")
        self.plotter.add_text("\n".join(lines), position="lower_edge", font_size=9,
                              color="black", name="info")

    # ====================================================================
    # Scale
    # ====================================================================
    @staticmethod
    def _nice_length(extent):
        """Largest 'round' scale length (mm) that is <= 30% of the extent being viewed."""
        for L in (50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1):
            if L <= 0.30 * extent:
                return L
        return 0.1

    def _add_scale_bar(self, bounds=None):
        """3-axis scale triad (readable from any viewing angle), sized to the current view."""
        b = bounds or self.body_bounds
        ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        L = self._nice_length(ext.max())
        corner = np.array([b[1] + 0.06 * ext.max(), b[2], b[4]])
        ends = []
        for i, axis in enumerate("xyz"):
            d = np.zeros(3)
            d[i] = L
            self.plotter.add_mesh(pv.Line(corner, corner + d), color="black", line_width=5,
                                  name=f"scale_{axis}", pickable=False, render=False)
            ends.append(corner + d + 0.02 * ext.max())
        txt = f"{L:g} mm"
        self.plotter.add_point_labels(ends, [txt] * 3, font_size=13, text_color="black",
                                      shape=None, show_points=False, always_visible=True,
                                      name="scale_label", render=False)

    def set_ruler(self, on):
        """mm ruler grid around the mouse (or around the isolated organ), counting from 0."""
        self._ruler_on = bool(on)
        self._draw_ruler()
        if self.ruler_box is not None:
            self.ruler_box.GetRepresentation().SetState(int(on))
        self.plotter.render()

    def _draw_ruler(self):
        self.plotter.remove_bounds_axes()
        if self._ruler_on:
            b = self._current_bounds()
            self.plotter.show_bounds(
                bounds=b, axes_ranges=(0, b[1] - b[0], 0, b[3] - b[2], 0, b[5] - b[4]),
                grid="back", location="outer", ticks="outside", xtitle="L-R (mm)",
                ytitle="Nose-Tail (mm)", ztitle="D-V (mm)", font_size=9, color="black",
                render=False)

    # ====================================================================
    # Slice planes
    # ====================================================================
    def _slice_actor_name(self, plane):
        return f"slice_{plane}"

    def _build_slice(self, plane, idx):
        s, sp, T = self.slice_step, self.slice_step * VOXEL_MM, VOXEL_MM
        h = sp / 2.0
        v = self.vol
        if plane == "coronal":
            arr = v[::s, idx, ::s]
            dims, spacing, origin = (arr.shape[0] + 1, 1, arr.shape[1] + 1), (sp, T, sp), (-h, idx * T, -h)
        elif plane == "sagittal":
            arr = v[idx, ::s, ::s]
            dims, spacing, origin = (1, arr.shape[0] + 1, arr.shape[1] + 1), (T, sp, sp), (idx * T, -h, -h)
        else:
            arr = v[::s, ::s, idx]
            dims, spacing, origin = (arr.shape[0] + 1, arr.shape[1] + 1, 1), (sp, sp, T), (-h, -h, idx * T)
        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
        lab2d = arr.T                                   # cell order: first array axis fastest
        grid.cell_data["label"] = lab2d.ravel().astype(np.float32)
        grid.cell_data["rgb"] = _LUT_U8[lab2d].reshape(-1, 3)
        return grid.threshold(0.5, scalars="label", preference="cell")   # drop background cells

    def _refresh_slice(self, plane):
        st = self.slices[plane]
        name = self._slice_actor_name(plane)
        if not st["show"]:
            self.plotter.remove_actor(name, render=False)
        else:
            m = self._build_slice(plane, st["index"])
            if m.n_cells == 0:
                self.plotter.remove_actor(name, render=False)
            else:
                self.plotter.add_mesh(m, scalars="rgb", rgb=True, lighting=False, name=name,
                                      reset_camera=False, pickable=False, render=False)
        self._update_clipping()
        self.plotter.render()

    def set_slice(self, plane, index=None, show=None, cut=None):
        """Set a plane's slice index (0-based), visibility and/or cut-away."""
        st = self.slices[plane]
        total = self.vol.shape[PLANES[plane]]
        if index is not None:
            st["index"] = int(np.clip(index, 0, total - 1))
        if show is not None:
            st["show"] = bool(show)
        if cut is not None:
            st["cut"] = bool(cut)
        self.active_plane = plane
        if self._ui_ready:
            self._update_slice_help()
        w = self.slice_widgets.get(plane)
        if w is not None:
            rep = w.GetRepresentation()
            rep.SetValue(st["index"])
            rep.SetTitleText(f"{PLANE_TITLES[plane]}  {st['index']} / {total}")
        self._sync_box(self.slice_show_boxes, plane, st["show"])
        self._sync_box(self.slice_cut_boxes, plane, st["cut"])
        self._refresh_slice(plane)

    def step_slice(self, delta, plane=None):
        plane = plane or self.active_plane
        self.set_slice(plane, index=self.slices[plane]["index"] + delta)

    def cycle_plane(self):
        names = list(PLANES)
        self.active_plane = names[(names.index(self.active_plane) + 1) % 3]
        self._update_slice_help()

    def flip_cut(self, plane=None):
        plane = plane or self.active_plane
        self.slices[plane]["side"] *= -1
        self._update_clipping()
        self.plotter.render()

    def _update_clipping(self):
        """Cut organs/skin away on one side of each plane that has cut-away enabled."""
        planes = []
        for p, ax in PLANES.items():
            st = self.slices[p]
            if st["cut"]:
                origin = [0.0, 0.0, 0.0]
                origin[ax] = st["index"] * VOXEL_MM
                normal = [0.0, 0.0, 0.0]
                normal[ax] = float(st["side"])          # geometry on the +normal side is kept
                vp = vtkPlane()
                vp.SetOrigin(*origin)
                vp.SetNormal(*normal)
                planes.append(vp)
        for actor in self.actors.values():
            mapper = actor.GetMapper()
            mapper.RemoveAllClippingPlanes()
            for vp in planes:
                mapper.AddClippingPlane(vp)

    def look_at_slice(self, plane=None):
        """Camera looks straight at the slice, from the cut-away side if cut is on."""
        plane = plane or self.active_plane
        ax, st = PLANES[plane], self.slices[plane]
        b = self.body_bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        center[ax] = st["index"] * VOXEL_MM
        normal = np.zeros(3)
        normal[ax] = 1.0
        side = -st["side"] if st["cut"] else -1.0       # camera on the removed side
        extent = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        up = (0, 1, 0) if plane == "axial" else (0, 0, 1)
        self.plotter.camera_position = [tuple(center + normal * side * extent * 1.6),
                                        tuple(center), up]
        foot = list(b)
        foot[2 * ax:2 * ax + 2] = [center[ax], center[ax]]      # footprint of the slice plane
        self.plotter.reset_camera(bounds=tuple(foot))
        self.plotter.camera.zoom(0.9)
        self._add_scale_bar(tuple(foot))             # scale sized to the slice you are looking at
        self.plotter.render()

    # ====================================================================
    # Widgets + input
    # ====================================================================
    @staticmethod
    def _sync_box(boxes, key, state):
        w = boxes.get(key)
        if w is not None:
            w.GetRepresentation().SetState(int(bool(state)))

    def _update_slice_help(self):
        W, H = self.window_size
        self.plotter.add_text(
            f"Active plane for keys: {self.active_plane}  (n = next)", position=(W - 440, 262),
            font_size=9, color="darkred", name="active_plane")

    def add_controls(self):
        p = self.plotter
        W, H = self.window_size

        # --- organ list: [show] [select] name ---
        x_show, x_sel, y0, dy = 10, 40, H - 70, 26
        p.add_text("show  sel", position=(x_show, H - 40), font_size=8, color="gray", name="hdr")
        for i, lab in enumerate(sorted(self.meshes)):
            y = y0 - i * dy
            self.vis_boxes[lab] = p.add_checkbox_button_widget(
                lambda state, l=lab: self.set_visible(l, state),
                value=True, position=(x_show, y), size=20, border_size=2,
                color_on=label_color(lab), color_off="lightgray", background_color="white")
            self.sel_boxes[lab] = p.add_checkbox_button_widget(
                lambda state, l=lab: self.select([l], add=True) if state
                else self._deselect(l),
                value=False, position=(x_sel, y), size=20, border_size=2,
                color_on="gold", color_off="white", background_color="lightgray")
            p.add_text(ORGAN_LABELS[lab], position=(x_sel + 30, y + 2), font_size=8,
                       color="black", name=f"lbl_{lab}")

        # --- isolate / clear / ruler ---
        y = 235
        self.ruler_box = p.add_checkbox_button_widget(
            self.set_ruler, value=False, position=(10, y), size=22, border_size=2,
            color_on="steelblue", color_off="white", background_color="lightgray")
        p.add_text("mm ruler grid", position=(40, y + 3), font_size=9, color="black", name="t_ruler")
        self.isolate_box = p.add_checkbox_button_widget(
            self.set_isolate, value=False, position=(10, y - 40), size=22, border_size=2,
            color_on="tomato", color_off="white", background_color="lightgray")
        p.add_text("Isolate selected", position=(40, y - 37), font_size=9, color="black", name="t_iso")

        def _clear(state):
            self.clear_selection()
            clear_box.GetRepresentation().SetState(0)
        clear_box = p.add_checkbox_button_widget(
            _clear, value=False, position=(10, y - 80), size=22, border_size=2,
            color_on="gold", color_off="white", background_color="lightgray")
        p.add_text("Clear selection", position=(40, y - 77), font_size=9, color="black", name="t_clr")

        # --- slice panel (bottom right): [show] [cut] slider ---
        x0 = W - 440
        p.add_text("show  cut", position=(x0 - 4, 232), font_size=8, color="gray", name="hdr2")
        for row, plane in enumerate(PLANES):
            y = 190 - row * 62
            total = self.vol.shape[PLANES[plane]]
            st = self.slices[plane]
            self.slice_show_boxes[plane] = p.add_checkbox_button_widget(
                lambda state, pl=plane: self.set_slice(pl, show=state),
                value=st["show"], position=(x0, y), size=20, border_size=2,
                color_on="seagreen", color_off="white", background_color="lightgray")
            self.slice_cut_boxes[plane] = p.add_checkbox_button_widget(
                lambda state, pl=plane: self.set_slice(pl, cut=state),
                value=st["cut"], position=(x0 + 45, y), size=20, border_size=2,
                color_on="purple", color_off="white", background_color="lightgray")
            self.slice_widgets[plane] = p.add_slider_widget(
                lambda v, pl=plane: self._on_slider(pl, v),
                [0, total - 1], value=st["index"],
                title=f"{PLANE_TITLES[plane]}  {st['index']} / {total}",
                pointa=((x0 + 95) / W, (y + 12) / H), pointb=((W - 30) / W, (y + 12) / H),
                style="modern", fmt="%.0f", title_height=0.015)
            self.slice_widgets[plane].GetRepresentation().ShowSliderLabelOff()  # title already shows the value

        # --- key bindings (avoid keys PyVista/VTK already use: q b v f w s r e p) ---
        p.add_key_event("a", self.show_all)
        p.add_key_event("h", self.hide_all)
        p.add_key_event("c", self.clear_selection)
        p.add_key_event("i", lambda: self.set_isolate(not self.isolate))
        p.add_key_event("n", self.cycle_plane)
        p.add_key_event("bracketleft", lambda: self.step_slice(-1))
        p.add_key_event("bracketright", lambda: self.step_slice(+1))
        p.add_key_event("comma", lambda: self.step_slice(-10))
        p.add_key_event("period", lambda: self.step_slice(+10))
        p.add_key_event("x", lambda: self.set_slice(self.active_plane,
                                                    cut=not self.slices[self.active_plane]["cut"]))
        p.add_key_event("t", self.flip_cut)
        p.add_key_event("g", self.look_at_slice)
        p.add_text(
            "Click organ = select (Shift = add)    i isolate    c clear    a/h show/hide all\n"
            "Slices:  n next plane   [ ] = -/+1   , . = -/+10   x cut-away   t flip cut   g look at slice",
            position="upper_edge", font_size=8, color="gray", name="help")
        self._update_slice_help()
        self._ui_ready = True

        # --- click-to-select (a click, not a drag, so rotating the view doesn't select) ---
        self._press_pos = None
        self._picker = vtkPropPicker()
        self._iren = p.iren.interactor      # NB: observer callbacks receive the *style*, not the interactor
        p.iren.add_observer("LeftButtonPressEvent", self._on_press)
        p.iren.add_observer("LeftButtonReleaseEvent", self._on_release)

    def _on_slider(self, plane, value):
        if self._ui_ready:
            self.set_slice(plane, index=int(round(value)), show=True)

    def _deselect(self, label):
        self.selected.discard(label)
        self._after_selection_change()

    def _on_press(self, caller, event):
        self._press_pos = self._iren.GetEventPosition()

    def _on_release(self, caller, event):
        if self._press_pos is None:
            return
        x, y = self._iren.GetEventPosition()
        px, py = self._press_pos
        self._press_pos = None
        if abs(x - px) <= self.CLICK_TOLERANCE_PX and abs(y - py) <= self.CLICK_TOLERANCE_PX:
            self.click_at(x, y, shift=bool(self._iren.GetShiftKey()))

    def click_at(self, x, y, shift=False):
        """Select the organ under screen position (x, y). Returns the label or None."""
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        actor = self._picker.GetActor()
        label = self._label_of_actor.get(id(actor)) if actor is not None else None
        if label is None:
            return None
        if shift:
            self.toggle_select(label)
        elif self.selected == {label}:
            self.clear_selection()
        else:
            self.select([label])
        return label

    # ====================================================================
    def show(self):
        self.add_controls()
        self.plotter.view_isometric()
        self.plotter.show()


# ---------------------------------------------------------------------------
# Run it (edit the path below, then press Run in VS Code)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    HDR_PATH = r"C:\Users\armen\Documents\_temp_matlab_R2024b_Windows\atlas\atlas\atlas_380x992x208.hdr"

    vol = load_analyze(HDR_PATH)
    print("Loaded", vol.shape, "- building meshes (takes a little while)...")
    meshes = build_all_meshes(vol, step=2)

    scene = OrganScene(vol, meshes)

    # Optional: start with a slice already showing, e.g. coronal 203 of 992
    # scene.set_slice("coronal", index=203, show=True, cut=True)

    scene.show()
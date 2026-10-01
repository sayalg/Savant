# Render the 5 carbonic anhydrase 2 panels for the benchmark figure.
#
#   cd report/benchmark_structures
#   pymol -cq render_panels.pml
#
# Writes panel_1.png .. panel_5.png, ordered by measured affinity (weakest
# first), all on ONE camera so the 5 are directly comparable. report/make_figures.py
# composes them with the predicted and measured values.
#
# Two things to know about these structures:
#   * They are Boltz-2 co-folded models, and none of them contains the catalytic
#     Zn(2+). Real sulfonamide binding to CA2 is zinc-coordinated, so the poses
#     cannot show that interaction. Worth a sentence in the caption.
#   * Use open-source PyMOL or a licensed copy. The evaluation build stamps
#     "No License File" across every ray-traced image.

reinitialize
bg_color white
set ray_opaque_background, 0
set opaque_background, 0

# ordered weakest -> strongest measured affinity
load v3_0195_model_0.pdb, p1
load v3_0305_model_0.pdb, p2
load v3_0095_model_0.pdb, p3
load v3_0075_model_0.pdb, p4
load v3_0388_model_0.pdb, p5

# superpose onto p1 so every panel shares one orientation
align p2, p1
align p3, p1
align p4, p1
align p5, p1

hide everything
show cartoon, polymer
show sticks, organic

# same protein in all 5 panels, so keep it neutral and let the ligand carry the
# colour. Colouring each protein differently implies 5 different targets.
color grey75, polymer
util.cnc("organic")
color tv_orange, organic and elem C

set cartoon_transparency, 0.45
set cartoon_fancy_helices, 1
set stick_radius, 0.22
set ray_shadows, 0
set antialias, 2
set ambient, 0.28
set specular, 0.25
set depth_cue, 0

# Pocket view. At 5-across in a 2-column paper each panel is ~1.4 in wide, and a
# whole-protein cartoon leaves the ligand a few pixels across. Zooming the site
# also makes the ligand size progression visible, which is part of the point.
orient (p1 and organic)
zoom (p1 and organic), 7

python
from pymol import cmd
view = cmd.get_view()
for i in range(1, 6):
    cmd.disable("all")
    cmd.enable("p%d" % i)
    cmd.set_view(view)
    cmd.ray(1100, 1100)
    cmd.png("panel_%d.png" % i, dpi=300)
print("wrote panel_1.png .. panel_5.png")
python end

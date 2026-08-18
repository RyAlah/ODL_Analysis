"""NPS explainer diagram in the slide-deck dark-navy theme."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc, Rectangle
import matplotlib.font_manager as fm

SURFACE = '#000026'
INK     = '#E7E7E7'
MUTED   = '#A9B2CE'
BAD     = '#E2534C'   # detractors
MID     = '#E0A83A'   # passives
GOOD    = '#18A566'   # promoters

_inst = {f.name for f in fm.fontManager.ttflist}
FONT  = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans') if f in _inst), 'DejaVu Sans')
plt.rcParams['font.family'] = FONT

fig, ax = plt.subplots(figsize=(12, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')


def box(x, y, w, h, edge, text, tcolor, fs=15, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.12',
                 linewidth=2, edgecolor=edge, facecolor=SURFACE))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            color=tcolor, fontsize=fs, fontweight='bold' if bold else 'normal')


# ── Formula row ─────────────────────────────────────────────────────────────
box(0.5, 4.7, 3.4, 0.9, MUTED, 'NET PROMOTER SCORE', INK, fs=14)
ax.text(4.25, 5.15, '=', ha='center', va='center', color=INK, fontsize=26, fontweight='bold')
box(4.7, 4.7, 3.0, 0.9, GOOD, '% PROMOTERS', GOOD, fs=14)
ax.text(8.05, 5.15, '–', ha='center', va='center', color=INK, fontsize=26, fontweight='bold')
box(8.5, 4.7, 3.0, 0.9, BAD, '% DETRACTORS', BAD, fs=14)


# ── Score squares 0..10 ─────────────────────────────────────────────────────
def face(cx, cy, mood):
    """mood: -1 sad, 0 neutral, 1 happy — drawn in muted ink."""
    for ex in (-0.12, 0.12):
        ax.add_patch(plt.Circle((cx + ex, cy + 0.10), 0.028, color=MUTED))
    if mood == 0:
        ax.plot([cx - 0.14, cx + 0.14], [cy - 0.13, cy - 0.13], color=MUTED, lw=2)
    else:
        th1, th2 = (200, 340) if mood == 1 else (20, 160)
        ax.add_patch(Arc((cx, cy - 0.13 + (0.10 if mood == 1 else -0.10)),
                         0.34, 0.30, angle=0, theta1=th1, theta2=th2, color=MUTED, lw=2))


sq, gap, y0 = 0.86, 0.18, 2.9
x = 0.7
for n in range(11):
    color = BAD if n <= 6 else MID if n <= 8 else GOOD
    ax.add_patch(FancyBboxPatch((x, y0), sq, sq,
                 boxstyle='round,pad=0,rounding_size=0.14',
                 linewidth=0, facecolor=color))
    ax.text(x + sq / 2, y0 + sq / 2, str(n), ha='center', va='center',
            color='white', fontsize=20, fontweight='bold')
    face(x + sq / 2, y0 - 0.62, -1 if n <= 6 else 0 if n <= 8 else 1)
    x += sq + gap

# bracket-label groups
groups = [(0, 6, 'DETRACTORS', BAD), (7, 8, 'PASSIVES', MID), (9, 10, 'PROMOTERS', GOOD)]
for lo, hi, label, color in groups:
    xl = 0.7 + lo * (sq + gap)
    xr = 0.7 + hi * (sq + gap) + sq
    ax.plot([xl, xr], [y0 - 1.05, y0 - 1.05], color=MUTED, lw=1.4)
    for xe in (xl, xr):
        ax.plot([xe, xe], [y0 - 1.05, y0 - 0.98], color=MUTED, lw=1.4)
    ax.text((xl + xr) / 2, y0 - 1.35, label, ha='center', va='center',
            color=color, fontsize=14, fontweight='bold')

plt.savefig('/home/user/ODL_Analysis/outputs/slide_theme/fig_nps_explainer.png',
            dpi=200, bbox_inches='tight', facecolor=SURFACE, pad_inches=0.25)
print('saved fig_nps_explainer.png')

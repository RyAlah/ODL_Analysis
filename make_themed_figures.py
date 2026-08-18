"""Render the analysis figures in the slide deck's dark-navy theme.

Companion to odl_sentiment_topic_analysis.ipynb. The notebook keeps its default
light styling; this script re-renders the same charts against the deck palette and
writes them to outputs/slide_theme/, leaving the light versions untouched.

    python make_themed_figures.py path/to/feedback_comparative_periods.xlsx

Palette provenance
------------------
Colours are taken from the deck's own theme part (ppt/theme/theme1.xml) and then
checked, not eyeballed:

  surface  #000026   theme lt1, the slide background
  ink      #E7E7E7   theme dk1, the body text colour
  blues    #005ACD / #307DE0 / #0100A2, the accent family

Every categorical set below was run through the dataviz skill's palette validator
against the #000026 surface. Two results are worth recording:

* The needs-difference scale (Little < Moderate < Large < Extreme) is ordinal, so
  it uses a single-hue ramp with monotonic lightness rather than a red-to-green
  one. A four-step red/amber/green ramp cannot be made CVD-safe - the middle steps
  collapse (worst adjacent pair reached only dE 2.9 under deuteranopia) - and a
  one-hue ramp also shows the ordering more directly. Anchoring runs light = more
  because the surface is dark.

* The genuinely categorical status colours (promoter/passive/detractor, yes/no,
  resolution) stay red/amber/green because the meaning is conventional. Amber
  cannot sit inside the validator's dark lightness band and stay separable from
  red - darkening it collapses the pair to dE 3.4 - so it is deliberately kept
  brighter than the band. Contrast against the surface still passes at >= 3:1, and
  every such chart carries category names plus value labels, which is the
  secondary encoding the floor-band CVD result requires.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

# ── Palette ────────────────────────────────────────────────────────────────
SURFACE    = '#000026'
INK        = '#E7E7E7'
INK_MUTED  = '#A9B2CE'
GRID       = '#1B1F4A'

PERIOD = {'focus': '#4A85E8', 'prior': '#21A896'}   # validated pair, dE 18.1 normal
SINGLE = '#4A85E8'
ANNOT  = '#E0A83A'

STATUS = {'good': '#18A566', 'mid': '#E0A83A', 'bad': '#E2534C'}
NPS_CAT = {'Promoter': STATUS['good'], 'Passive': STATUS['mid'], 'Detractor': STATUS['bad']}
RESOLUTION = {'Positive': STATUS['good'], 'Neutral': SINGLE, 'Negative': STATUS['bad']}

ORDINAL = ['#42699F', '#4A7FD4', '#6B9FF2', '#A5C9F7']   # monotonic L, all >= 3:1

OUT = Path('outputs/slide_theme')

# The deck sets Arial. Where it is absent, fall back to Liberation Sans, which is
# metric-compatible, rather than leaving Arial first and emitting a findfont warning
# for every text object drawn.
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
FONT_STACK = [f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans') if f in _installed]

plt.rcParams.update({
    'figure.facecolor':  SURFACE,
    'axes.facecolor':    SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family':       FONT_STACK,
    'font.size':         11,
    'figure.dpi':        150,
    'text.color':        INK,
    'axes.labelcolor':   INK,
    'axes.titlecolor':   INK,
    'xtick.color':       INK_MUTED,
    'ytick.color':       INK_MUTED,
    'axes.edgecolor':    GRID,
    'axes.grid':         True,
    'grid.color':        GRID,
    'grid.linewidth':    0.8,
    'legend.facecolor':  SURFACE,
    'legend.edgecolor':  GRID,
    'legend.labelcolor': INK,
})


def tidy(ax, grid_axis='y'):
    """Recede the frame: no top/right spines, grid on one axis only, behind marks."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {OUT / name}')


def ink_on(hex_color):
    """Readable label colour for text sitting on a filled mark."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return SURFACE if lum > 0.4 else INK


# ── Load ───────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_excel(path).rename(columns={
        'Client Feedback: Created Date':           'date',
        'Case: Degree of Resolution':              'resolution',
        'Case: Case Owner':                        'case_owner',
        'Net Promoter Score':                      'nps',
        'How much of a positive diff has ODL had': 'positive_diff',
        'Informed About Case':                     'informed',
        'Did we give choices':                     'gave_choices',
        'Barriers to services':                    'barriers',
        'Anything we could not have helped with?': 'beyond_scope',
        'Needs Met (numeric)':                     'needs_met_num',
        'Needs Met (source)':                      'needs_met_source',
        'Positive Diff (numeric)':                 'positive_diff_num',
        'NPS Category':                            'nps_category',
        'Gave Choices (1=Yes)':                    'gave_choices_bin',
        'Barriers Reported (1=Yes)':               'barriers_bin',
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['needs_scale'] = np.where(
        df['needs_met_source'].astype(str).str.startswith('current'), 'current', 'legacy')
    return df


# ── Figures ────────────────────────────────────────────────────────────────
def fig_nps_focus(df, foc):
    nps = ((foc['nps_category'] == 'Promoter').mean()
           - (foc['nps_category'] == 'Detractor').mean()) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    counts = foc['nps'].value_counts().reindex(range(11), fill_value=0)
    band = {**{i: STATUS['bad'] for i in range(0, 7)},
            **{i: STATUS['mid'] for i in range(7, 9)},
            **{i: STATUS['good'] for i in range(9, 11)}}
    axes[0].bar(counts.index, counts.values,
                color=[band[i] for i in counts.index], width=0.82)
    axes[0].set_title('Score distribution', fontsize=12, pad=10)
    axes[0].set_xlabel('Net Promoter Score (0–10)')
    axes[0].set_ylabel('Clients')
    axes[0].set_xticks(range(11))
    tidy(axes[0])

    cats = ['Promoter', 'Passive', 'Detractor']
    vals = [(foc['nps_category'] == c).sum() for c in cats]
    bars = axes[1].bar(cats, vals, color=[NPS_CAT[c] for c in cats], width=0.6)
    axes[1].bar_label(bars, fmt='%d', padding=4, color=INK, fontsize=11)
    axes[1].set_title(f'Categories — NPS = {nps:.0f}', fontsize=12, pad=10)
    axes[1].set_ylabel('Clients')
    axes[1].set_ylim(0, max(vals) * 1.15)
    tidy(axes[1])

    fig.suptitle(f'Net Promoter Score — focus period (n={len(foc)})',
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_nps_focus.png')


def fig_nps_by_period(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.4
    for i, p in enumerate(['prior', 'focus']):
        g = df[df['period'] == p]
        c = g['nps'].value_counts().reindex(range(11), fill_value=0)
        pct = c / c.sum() * 100
        ax.bar(np.array(range(11)) + (i - 0.5) * w, pct, w,
               color=PERIOD[p], label=f'{p}  (n={len(g)}, mean {g["nps"].mean():.2f})')
    ax.set_xticks(range(11))
    ax.set_xlabel('Net Promoter Score (0–10)')
    ax.set_ylabel('% of responses in period')
    ax.set_title('NPS distribution by period', fontweight='bold', fontsize=14, pad=12)
    ax.legend(frameon=False, labelcolor=INK)
    tidy(ax)
    fig.tight_layout()
    save(fig, 'fig_nps_by_period.png')


def fig_ratings_focus(df, foc, pd_order):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    vc = foc['positive_diff'].value_counts()
    vals = [vc.get(o, 0) for o in pd_order]
    bars = axes[0].bar(pd_order, vals,
                       color=ORDINAL[:len(pd_order)], width=0.62)
    axes[0].bar_label(bars, fmt='%d', padding=4, color=INK, fontsize=11)
    axes[0].set_title('How much of a positive difference\nhas ODL had?', fontsize=12, pad=10)
    axes[0].set_ylabel('Clients')
    axes[0].set_ylim(0, max(vals) * 1.15)
    tidy(axes[0])

    res = foc['resolution'].value_counts().dropna()
    bars = axes[1].bar(res.index, res.values,
                       color=[RESOLUTION.get(r, SINGLE) for r in res.index], width=0.55)
    axes[1].bar_label(bars, fmt='%d', padding=4, color=INK, fontsize=11)
    axes[1].set_title('Degree of resolution', fontsize=12, pad=10)
    axes[1].set_ylabel('Clients')
    axes[1].set_ylim(0, res.max() * 1.15)
    tidy(axes[1])

    fig.suptitle(f'Client satisfaction ratings — focus period (n={len(foc)})',
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_ratings_focus.png')


def fig_yesno_focus(df, foc):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))

    hist = foc['informed'].dropna()
    axes[0].hist(hist, bins=np.arange(0, 105, 5), color=SINGLE)
    axes[0].set_title(f'Informed about case\nmedian {hist.median():.0f}, '
                      f'{100 * (hist >= 90).mean():.0f}% ≥ 90', fontsize=10, pad=8)
    axes[0].set_xlabel('Score (0–100)')
    axes[0].set_ylabel('Clients')
    tidy(axes[0])

    # Colour encodes good/bad, not the literal string. "Yes" is the desirable answer
    # for gave-choices but the undesirable one for barriers and beyond-scope, so a
    # blanket Yes=green rule would paint reported barriers as a positive result.
    good_answer = {'Gave choices': 'Yes', 'Barriers to service': 'No', 'Beyond scope': 'No'}

    for ax, (label, col) in zip(axes[1:], [
            ('Gave choices', foc['gave_choices']),
            ('Barriers to service', foc['barriers']),
            ('Beyond scope', foc['beyond_scope'])]):
        counts = col.value_counts()
        good = good_answer[label]
        cols = [STATUS['good'] if i == good else STATUS['bad'] if i in ('Yes', 'No') else SINGLE
                for i in counts.index]
        bars = ax.bar(counts.index, counts.values, color=cols, width=0.55)
        ax.bar_label(bars, fmt='%d', padding=4, color=INK, fontsize=10)
        pct = counts / counts.sum() * 100
        ax.set_title(f'{label}\n{pct.get(good, 0):.0f}% {good}  (favourable)',
                     fontsize=10, pad=8)
        ax.set_ylim(0, counts.max() * 1.18)
        tidy(ax)

    fig.suptitle(f'Response distributions — focus period (n={len(foc)})',
                 fontweight='bold', fontsize=15, y=1.04)
    fig.tight_layout()
    save(fig, 'fig_yesno_focus.png')


def fig_nps_by_owner_focus(df, foc):
    counts = foc['case_owner'].value_counts()
    active = counts[counts >= 5].index
    owner = (foc[foc['case_owner'].isin(active)]
             .groupby('case_owner')['nps'].agg(['mean', 'count'])
             .sort_values('mean', ascending=True))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(owner.index, owner['mean'], color=SINGLE, height=0.68)
    mean = foc['nps'].mean()
    ax.axvline(mean, color=ANNOT, linestyle='--', linewidth=1.4,
               label=f'Focus-period mean ({mean:.1f})')

    # Labels sit in a fixed column past the longest bar. Tucking them just after each
    # bar end instead would run them straight through the mean line for every owner
    # scoring near the average.
    LABEL_X = 10.35
    for i, (_, row) in enumerate(owner.iterrows()):
        ax.text(LABEL_X, i, f"{row['mean']:.1f}   n={int(row['count'])}",
                va='center', fontsize=8.5, color=INK_MUTED)

    ax.set_xlim(0, 12.4)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel('Mean NPS')
    ax.set_title(f'Average NPS by case owner — focus period\n'
                 f'{len(active)} owners with 5+ responses',
                 fontweight='bold', fontsize=13, pad=32)
    ax.legend(frameon=False, labelcolor=INK,
              loc='lower left', bbox_to_anchor=(0, 1.005))
    tidy(ax, grid_axis='x')
    fig.tight_layout()
    save(fig, 'fig_nps_by_owner_focus.png')


def fig_scale_effect(df):
    foc = df[df['period'] == 'focus']
    cur = foc.loc[foc['needs_scale'] == 'current', 'needs_met_num'].dropna().values
    leg = foc.loc[foc['needs_scale'] == 'legacy', 'needs_met_num'].dropna().values
    lp = df[(df['needs_scale'] == 'legacy') & (df['period'] == 'prior')]['needs_met_num'].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    xs, w = np.arange(1, 6), 0.38
    axes[0].bar(xs - w / 2, [100 * (leg == v).mean() for v in xs], w,
                color=PERIOD['focus'], label=f'quality wording (n={len(leg)})')
    axes[0].bar(xs + w / 2, [100 * (cur == v).mean() for v in xs], w,
                color=ANNOT, label=f'agreement wording (n={len(cur)})')
    axes[0].set_xticks(xs)
    axes[0].set_xlabel('Response (1 = worst, 5 = best)')
    axes[0].set_ylabel('% of responses')
    axes[0].set_title('Same period, different wording', fontsize=12, pad=10)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=9)
    tidy(axes[0])

    groups = [('prior\nquality', lp, PERIOD['focus']),
              ('focus\nquality', leg, PERIOD['focus']),
              ('focus\nagreement', cur, ANNOT)]
    means = [g.mean() for _, g, _ in groups]
    errs = [1.96 * g.std(ddof=1) / np.sqrt(len(g)) for _, g, _ in groups]
    axes[1].bar([g[0] for g in groups], means, yerr=errs, capsize=6, width=0.55,
                color=[g[2] for g in groups],
                error_kw={'ecolor': INK_MUTED, 'elinewidth': 1.3})
    for i, (m, e) in enumerate(zip(means, errs)):
        axes[1].text(i, m + e + 0.09, f'{m:.2f}', ha='center', fontsize=11, color=INK)
    axes[1].set_ylabel('Mean needs-met score (1–5)')
    axes[1].set_ylim(0, 5.6)
    axes[1].set_title('The jump tracks the wording, not the period', fontsize=12, pad=10)
    tidy(axes[1])

    fig.suptitle('Needs met — scale change artefact', fontweight='bold', fontsize=15, y=1.03)
    fig.tight_layout()
    save(fig, 'fig_scale_effect.png')


def fig_positive_diff_stacked(df, pd_order):
    a = df.loc[df['period'] == 'focus', 'positive_diff_num'].dropna().values
    b = df.loc[df['period'] == 'prior', 'positive_diff_num'].dropna().values
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 2 * u / (len(a) * len(b)) - 1

    ct = (pd.crosstab(df['period'], df['positive_diff'], normalize='index') * 100)
    ct = ct[pd_order].reindex(['prior', 'focus'])
    assert np.allclose(ct.sum(axis=1), 100)

    fig, ax = plt.subplots(figsize=(11.5, 3.3))
    left = np.zeros(len(ct))
    for cat, color in zip(ct.columns, ORDINAL):
        vals = ct[cat].values
        ax.barh(ct.index, vals, left=left, color=color, height=0.62,
                edgecolor=SURFACE, linewidth=2, label=cat)   # 2px surface gap
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 3.5:   # 3.5 keeps the 4% focus/Little segment labelled like its 5% peer
                ax.text(l + v / 2, i, f'{v:.0f}%', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=ink_on(color))
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel('% of responses')
    ax.set_title('"How much of a positive difference has ODL had?"\n'
                 f'Mann–Whitney p = {p:.3f},  rank-biserial r = {r:+.3f}',
                 fontweight='bold', fontsize=12.5, pad=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False,
              labelcolor=INK, fontsize=9.5)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis='y', length=0, labelcolor=INK)
    fig.tight_layout()
    save(fig, 'fig_positive_diff_stacked.png')


def fig_proportions(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (col, label) in zip(axes, [('gave_choices_bin', 'Gave choices'),
                                       ('barriers_bin', 'Barriers reported')]):
        a = df.loc[df['period'] == 'focus', col].dropna()
        b = df.loc[df['period'] == 'prior', col].dropna()
        s1, n1, s2, n2 = int(a.sum()), len(a), int(b.sum()), len(b)
        p1, p2 = s1 / n1, s2 / n2
        z, pv = proportions_ztest([s1, s2], [n1, n2])
        lo, hi = confint_proportions_2indep(s1, n1, s2, n2, compare='diff', method='newcomb')

        bars = ax.bar(['prior', 'focus'], [p2 * 100, p1 * 100],
                      color=[PERIOD['prior'], PERIOD['focus']], width=0.5)
        ax.bar_label(bars, fmt='%.1f%%', padding=5, color=INK, fontsize=11)
        ax.set_ylabel('% Yes')
        ax.set_ylim(0, max(p1, p2) * 100 * 1.3)
        ax.set_title(f'{label}\n{(p1 - p2) * 100:+.1f} pp   '
                     f'95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]   p = {pv:.3f}',
                     fontsize=11, pad=10)
        tidy(ax)

    fig.suptitle('Binary measures — focus vs prior', fontweight='bold', fontsize=15, y=1.04)
    fig.tight_layout()
    save(fig, 'fig_proportions.png')


def fig_informed(df):
    a = df.loc[df['period'] == 'focus', 'informed'].dropna().values
    b = df.loc[df['period'] == 'prior', 'informed'].dropna().values
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.arange(0, 105, 5)
    axes[0].hist([b, a], bins=bins, label=['prior', 'focus'],
                 color=[PERIOD['prior'], PERIOD['focus']])
    axes[0].set_xlabel('Informed about case (0–100)')
    axes[0].set_ylabel('Clients')
    axes[0].set_title('Ceiling-piled, with a separate cluster at 0', fontsize=12, pad=10)
    axes[0].legend(frameon=False, labelcolor=INK)
    tidy(axes[0])

    labels = ['median', '% ≥ 90', '% = 0']
    pv = [np.median(b), 100 * (b >= 90).mean(), 100 * (b == 0).mean()]
    fv = [np.median(a), 100 * (a >= 90).mean(), 100 * (a == 0).mean()]
    x, w = np.arange(3), 0.38
    axes[1].bar(x - w / 2, pv, w, color=PERIOD['prior'], label='prior')
    axes[1].bar(x + w / 2, fv, w, color=PERIOD['focus'], label='focus')
    for i, (p_, f_) in enumerate(zip(pv, fv)):
        axes[1].text(i - w / 2, p_ + 1.5, f'{p_:.0f}', ha='center', fontsize=9.5, color=INK)
        axes[1].text(i + w / 2, f_ + 1.5, f'{f_:.0f}', ha='center', fontsize=9.5, color=INK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 112)
    axes[1].set_title(f'Robust summaries — Mann–Whitney p = {p:.3f}', fontsize=12, pad=10)
    axes[1].legend(frameon=False, labelcolor=INK)
    tidy(axes[1])

    fig.suptitle('Informed about case', fontweight='bold', fontsize=15, y=1.03)
    fig.tight_layout()
    save(fig, 'fig_informed.png')


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'feedback_comparative_periods.xlsx'
    df = load(src)
    foc = df[df['period'] == 'focus'].copy()
    pd_order = (df.dropna(subset=['positive_diff', 'positive_diff_num'])
                  .groupby('positive_diff')['positive_diff_num'].first()
                  .sort_values().index.tolist())

    print(f'Loaded {len(df)} responses; focus period n={len(foc)}')
    print(f'Positive-difference scale: {" < ".join(pd_order)}')

    fig_nps_focus(df, foc)
    fig_nps_by_period(df)
    fig_ratings_focus(df, foc, pd_order)
    fig_yesno_focus(df, foc)
    fig_nps_by_owner_focus(df, foc)
    fig_scale_effect(df)
    fig_positive_diff_stacked(df, pd_order)
    fig_proportions(df)
    fig_informed(df)
    print('done')


if __name__ == '__main__':
    main()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np, pandas as pd

SRC='/root/.claude/uploads/8233d335-e81d-51ef-8c70-4e86a921efcc/eb178790-feedback_comparative_periods.xlsx'
df=pd.read_excel(SRC)
df['date']=pd.to_datetime(df['Client Feedback: Created Date'],errors='coerce')
df=df.dropna(subset=['Informed About Case','date'])

g=df.set_index('date').resample('ME')['Informed About Case'].agg(
    n='count', p90=lambda s:100*(s>=90).mean(), p0=lambda s:100*(s==0).mean())
FOCUS_START=pd.Timestamp('2025-06-01')
MIN_N=8
g_full=g[g.n>=MIN_N]; g_thin=g[g.n<MIN_N]

def render(themed, fname):
    if themed:
        surface,ink,muted,grid='#000026','#E7E7E7','#A9B2CE','#1B1F4A'
        c90,c0,shade='#4A85E8','#E2534C','#0A1040'
    else:
        surface,ink,muted,grid='white','black','dimgray','#d9d9d9'
        c90,c0,shade='steelblue','#c0392b','#eef2fb'

    fig,axes=plt.subplots(1,2,figsize=(14,4.8),facecolor=surface)
    panels=[(axes[0],'p90','% of clients scoring ≥ 90',c90,'Well-informed clients'),
            (axes[1],'p0','% of clients scoring 0',c0,'Clients who felt uninformed (score 0)')]
    for ax,col,ylab,color,title in panels:
        ax.set_facecolor(surface)
        ax.axvspan(FOCUS_START,g.index.max(),color=shade,zorder=0)
        ax.plot(g_full.index,g_full[col],color=color,lw=2,zorder=3)
        ax.scatter(g_full.index,g_full[col],s=np.clip(g_full.n*2.2,20,260),
                   color=color,edgecolor=surface,linewidth=1.2,zorder=4)
        if len(g_thin):
            ax.scatter(g_thin.index,g_thin[col],s=40,facecolor='none',
                       edgecolor=muted,linewidth=1.2,zorder=4)
        mean=100*((df['Informed About Case']>=90).mean() if col=='p90'
                  else (df['Informed About Case']==0).mean())
        ax.axhline(mean,color=muted,ls='--',lw=1,zorder=2,
                   label=f'Overall {mean:.0f}%')
        ymax=max(g_full[col].max()*1.15,mean*1.3)
        ax.text(FOCUS_START+(g.index.max()-FOCUS_START)/2, ymax*0.96,'focus period',
                ha='center',va='top',color=muted,fontsize=9,style='italic')
        ax.set_ylim(0,ymax); ax.set_ylabel(ylab,color=ink)
        ax.set_title(title,color=ink,fontsize=12,pad=8)
        ax.legend(frameon=False,labelcolor=ink,loc='upper right',fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(colors=muted); plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
        for s in ('top','right'): ax.spines[s].set_visible(False)
        for s in ('left','bottom'): ax.spines[s].set_color(grid)
        ax.set_axisbelow(True); ax.grid(True,axis='y',color=grid,lw=0.8)
        for gl in ax.get_ygridlines(): gl.set_color(grid)

    fig.suptitle('Informed About Case — Monthly Trend',color=ink,
                 fontweight='bold',fontsize=15,y=1.02)
    fig.text(0.5,-0.06,'Marker size ∝ responses that month; hollow markers are months with '
             f'fewer than {MIN_N} responses. Aug 2026 is a partial month.',
             ha='center',color=muted,fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fname,dpi=150,bbox_inches='tight',facecolor=surface)
    plt.close(fig); print('saved',fname)

import os
os.makedirs('/home/user/ODL_Analysis/outputs/slide_theme',exist_ok=True)
render(False,'/home/user/ODL_Analysis/outputs/fig_informed_trend.png')
render(True ,'/home/user/ODL_Analysis/outputs/slide_theme/fig_informed_trend.png')

"""Plot only the newly executed population-structure sensitivity."""
from pathlib import Path
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({'font.family':'Times New Roman','font.size':10,
                     'axes.labelsize':10,'axes.titlesize':10.5,'xtick.labelsize':9,
                     'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
                     'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
                     'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
                     'mathtext.sf':'Times New Roman','mathtext.fallback':None,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.7})
COLORS = ['#4C91C9','#299E7C','#9662B8']
BUDGETS = [240,480,720]
DATA = json.loads((HERE/'results.json').read_text())
OUT = HERE.parent/'figures'
OUT.mkdir(exist_ok=True)


def endpoints(budget):
    a=sorted([r for r in DATA['endpoints'] if r['budget']==budget and r['stage']==0],key=lambda r:r['seed'])
    b=sorted([r for r in DATA['endpoints'] if r['budget']==budget and r['stage']==4],key=lambda r:r['seed'])
    assert len(a)==len(b)==8
    return a,b


def paired(ax,key):
    for j,(budget,color) in enumerate(zip(BUDGETS,COLORS)):
        a,b=endpoints(budget)
        values=np.array([[r['count_decomposition'][key] for r in rows] for rows in (a,b)])
        positions=np.array([j*3,j*3+1])
        jitter=np.linspace(-.12,.12,8)
        for i in range(8):
            ax.plot(positions+jitter[i],values[:,i],color=color,alpha=.45,lw=.65,zorder=1)
        for k in range(2):
            ax.scatter(positions[k]+jitter,values[k],s=14,edgecolors=color,
                       facecolors='white' if k==0 else color,lw=.65,zorder=3)
            ax.errorbar(positions[k],values[k].mean(),yerr=values[k].std(ddof=1),
                        color='black',fmt='D',ms=3,lw=.8,capsize=2,zorder=4)
    ax.set_xticks([.5,3.5,6.5],['3.2','6.4','9.6'])
    ax.set_xlabel('Activity target (spikes/s/cell)')
    ax.set_xlim(-.5,7.5)
    ax.set_ylim(bottom=0)


def dependence_sensitivity(ax,indices,xlabels,xlabel):
    for j,(budget,color) in enumerate(zip(BUDGETS,COLORS)):
        points=[]
        for x,index in enumerate(indices):
            a,b=endpoints(budget)
            delta=np.array([v['diagnostics'][index]['signed_excess_bits']-u['diagnostics'][index]['signed_excess_bits'] for u,v in zip(a,b)])
            position=x+(j-1)*.08
            ax.scatter(position+np.linspace(-.025,.025,8),delta,s=9,color=color,alpha=.28,lw=0)
            ax.errorbar(position,delta.mean(),yerr=delta.std(ddof=1),color=color,
                        fmt='o',ms=3.5,lw=.8,capsize=2)
            points.append((position,delta.mean()))
        ax.plot(*np.array(points).T,color=color,lw=.8)
    ax.axhline(0,color='.6',ls=':',lw=.7)
    ax.set_xticks(range(len(indices)),xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Change in TC excess\n(bits/group-window)')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,-2),useMathText=True)


def main():
    fig,axs=plt.subplots(2,3,figsize=(8.4,5.4))
    fig.subplots_adjust(left=.09,right=.985,bottom=.105,top=.88,wspace=.54,hspace=.64)
    for ax,label,title in zip(axs.flat,'ABCDEF',[
        'Cell heterogeneity (1 s)','Mean marginal gap (1 s)','Dependence (four cells, 20 ms)',
        'Group-size sensitivity','Window sensitivity','Repeat sensitivity']):
        ax.text(-.23,1.14,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
        ax.set_title(title,pad=11)
        ax.tick_params(length=3,width=.6)
    paired(axs[0,0],'cell_heterogeneity_bits')
    axs[0,0].set_ylabel(r'Cell-identity information, $J_{\rm cell}$'+'\n(bits)')
    paired(axs[0,1],'mean_marginal_gap_per_spike')
    axs[0,1].set_ylabel('Mean marginal gap\n(bits/spike)')
    ax=axs[0,2]
    for j,(budget,color) in enumerate(zip(BUDGETS,COLORS)):
        rows=endpoints(budget)
        x=np.array([0,1])+j*3
        for key,style,marker in [('observed_TC_bits','-','o'),('conditional_shuffle_TC_bits','--','s')]:
            vals=np.array([[r['diagnostics'][0][key] for r in stage] for stage in rows])
            ax.errorbar(x,vals.mean(axis=1),yerr=vals.std(axis=1,ddof=1),color=color,
                        ls=style,lw=.9,capsize=2)
            ax.scatter(x[:1],vals.mean(axis=1)[:1],s=16,edgecolors=color,facecolors='white',zorder=4,lw=.8)
            ax.scatter(x[1:],vals.mean(axis=1)[1:],s=16,edgecolors=color,facecolors=color,zorder=4,lw=.8)
    ax.set_xticks([.5,3.5,6.5],['3.2','6.4','9.6'])
    ax.set_xlabel('Activity target (spikes/s/cell)')
    ax.set_ylabel('Total correlation\n(bits/group-window)')
    ax.set_ylim(bottom=0)
    ax.legend(handles=[Line2D([],[],color='black',ls='-',label='Observed'),
                       Line2D([],[],color='black',ls='--',label='Conditional shuffle')],
              frameon=False,fontsize=8,loc='upper left',handlelength=1.4)
    dependence_sensitivity(axs[1,0],[1,0,2],['2','4','8'],'Neurons per group (20 ms)')
    dependence_sensitivity(axs[1,1],[0,3,4],['20','50','100'],'Window (ms; four cells)')
    dependence_sensitivity(axs[1,2],[5,0],['4','8'],'Repeats (four cells, 20 ms)')
    handles=[Line2D([],[],color=c,marker='o',lw=.9,ms=4,label=f'{b/75:g}') for b,c in zip(BUDGETS,COLORS)]
    handles += [Line2D([],[],color='black',marker='o',mfc='white',ls='none',ms=4,label='Initial'),
                Line2D([],[],color='black',marker='o',ls='none',ms=4,label='Learned')]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.55,1.015),frameon=False,
               title='Activity target (spikes/s/cell)',fontsize=9,title_fontsize=10,
               handlelength=1.5,columnspacing=1.3)
    pdf=OUT/'Appendix_1_figure_14_NEW.pdf'
    fig.savefig(pdf,metadata={'Title':'Population structure sensitivity','Subject':'New analysis of archived 24-network responses'})
    fig.savefig(OUT/'Appendix_1_figure_14_NEW.png',dpi=220)
    plt.close(fig)
    target=ROOT/'output/overleaf/GibbsManifold_eLife_Reconsideration_20261109/figures/Appendix_1_figure_14.pdf'
    shutil.copy2(pdf,target)
    print(pdf)


if __name__=='__main__':
    main()

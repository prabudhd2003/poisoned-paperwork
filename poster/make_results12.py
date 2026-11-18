# Section 5 chart: every defense on the same 396 attacked receipts (from evaluation_summary.csv).
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Patch
names={f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({'font.family':next(n for n in ['TeX Gyre Heros','DejaVu Sans'] if n in names),'font.size':21})
GOOD='#2A7F62';BAD='#B3261E';CARD='#990000';MUTE='#666666';REJ='#E6E6E6'
G=[('No defense',[('Qwen2.5-VL alone',64,332,0,313)]),
   ('Clean the image first',[('JPEG',115,281,0,260),('Resize + vote',94,302,0,286),('Vote over views',76,313,7,299),('Re-worded questions',78,282,36,272)]),
   ('Refuse unless sources agree',[('OCR only',106,3,287,0),('Model + OCR + crop',22,0,374,0),('Two bands agree',60,0,336,0),('Masked-band check',49,0,347,0)]),
   ('Read a crop',[('Fixed crop',313,83,0,0),('One located band',356,28,12,0)]),
   ('Ours',[('CBVR',354,7,35,0)])]
fig,ax=plt.subplots(figsize=(15,8.6))
y=0;yt=[];yl=[];XR=432
for gi,(g,ms) in enumerate(G):
    if gi: y+=.55
    ax.text(0,y,g.upper(),fontsize=17,color=MUTE,fontweight='bold',va='center',ha='left')
    y+=.85
    for lab,cr,wr,rj,tg in ms:
        ours=lab=='CBVR'
        if ours: ax.axhspan(y-.48,y+.48,xmin=-.33,xmax=1,color='#FBEFD0',zorder=0,clip_on=False)
        h=.6
        ax.barh(y,cr,color=GOOD,height=h,edgecolor='white',lw=2,zorder=2)
        ax.barh(y,wr,left=cr,color=BAD,height=h,edgecolor='white',lw=2,zorder=2)
        ax.barh(y,rj,left=cr+wr,color=REJ,height=h,edgecolor='white',lw=2,hatch='//',zorder=2)
        if cr>40: ax.text(cr-7,y,str(cr),ha='right',va='center',color='white',fontsize=19,fontweight='bold',zorder=3)
        else: ax.text(cr+5,y,str(cr),ha='left',va='center',color=GOOD,fontsize=19,fontweight='bold',zorder=3)
        if wr>40: ax.text(cr+wr-7,y,str(wr),ha='right',va='center',color='white',fontsize=19,zorder=3)
        if rj>40: ax.text(cr+wr+rj-7,y,str(rj),ha='right',va='center',color='#777777',fontsize=19,zorder=3)
        ax.text(XR,y,str(tg),ha='center',va='center',fontsize=22,color=BAD if tg else GOOD,fontweight='bold')
        yt.append(y);yl.append(lab);y+=1
ax.set_yticks(yt);ax.set_yticklabels(yl,fontsize=21);ax.set_ylim(y-.2,-1.3);ax.set_xlim(0,450)
for t,l in zip(ax.get_yticklabels(),yl):
    if l=='CBVR':t.set_color(CARD);t.set_fontweight('bold');t.set_fontsize(23)
ax.set_xticks([0,100,200,300,396]);ax.tick_params(axis='y',length=0,pad=10)
for s in ('left','top','right'):ax.spines[s].set_visible(False)
ax.spines['bottom'].set_bounds(0,396)
for x in (100,200,300,396):ax.axvline(x,color='#EEEEEE',lw=1.2,zorder=0)
ax.plot([414,414],[-1.1,y-.5],color='#DDDDDD',lw=1.5,clip_on=False)
ax.text(XR,-.95,"attacker's\ntotal",ha='center',va='center',fontsize=17,color=BAD,fontweight='bold',linespacing=1.0)
ax.set_xlabel('396 attacked images of 44 held-out receipts',color=MUTE,labelpad=8)
ax.legend(handles=[Patch(color=GOOD,label='right total'),Patch(color=BAD,label='wrong total accepted'),Patch(facecolor=REJ,hatch='//',label='sent to a person')],
          frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(-.01,1.0),fontsize=20,handlelength=1.3,columnspacing=1.6)
fig.savefig('figures/c_results12.png',dpi=200,bbox_inches='tight');plt.close(fig)

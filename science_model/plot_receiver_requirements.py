"""Plot receiver budget scenarios; lines only connect solved frequency samples."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
rows=list(csv.DictReader((ROOT/'results/receiver_requirements/comparison.csv').open()))
thermal=list(csv.DictReader((ROOT/'results/receiver_requirements/passive_temperature_and_correlation_screen.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
for case,label,color in [('reference','35 pF, 3 nV', '#bb4433'),('capacitance_only','10 pF, 3 nV','#d38f20'),('prototype_target','3 pF, 0.7 nV, 0.1 pA','#126c96')]:
    r=[x for x in rows if x['case']==case and x['band']=='HF' and x['sky_scale']=='0.3' and float(x['frequency_mhz'])>=10]
    axes[0].semilogy([float(x['frequency_mhz']) for x in r],[float(x['worst_nonsky_over_sky']) for x in r],'-o',ms=4,label=label,color=color)
r=[x for x in thermal if x['ground_temperature_k']=='100' and x['bias_and_series_temperature_c']=='125']
axes[0].semilogy([float(x['frequency_mhz']) for x in r],[float(x['largest_tested_nonsky_over_sky']) for x in r],'--',color='#126c96',label='Target + protection/correlation stress')
axes[0].axhline(.25,color='black',ls=':',label='4:1 sky / non-sky goal')
axes[0].set(xlabel='Frequency (MHz)',ylabel='Non-sky noise / sky noise power',title='HF pair: receiver budget comparison')
axes[0].legend(fontsize=8,frameon=False,loc='upper left')
for case,label,color in [('screen_3pf_1nv','3 pF','#126c96'),('screen_5pf_1nv','5 pF','#159b70'),('capacitance_only','10 pF (zero current noise)','#d38f20')]:
    r=[x for x in rows if x['case']==case and x['band']=='HF' and x['sky_scale']=='0.3' and float(x['frequency_mhz'])>=10]
    axes[1].plot([float(x['frequency_mhz']) for x in r],[float(x['en_limit_nv_at_given_in']) for x in r],'-o',ms=4,label=label,color=color)
axes[1].axhline(.7,color='black',ls=':',label='0.7 nV target')
axes[1].set(xlabel='Frequency (MHz)',ylabel='Allowed differential voltage noise (nV/sqrt(Hz))',title='HF pair: voltage-noise budget',ylim=(0,4))
axes[1].legend(fontsize=8,frameon=False,loc='upper right')
for ax in axes:
    ax.axvspan(14,46,color='#126c96',alpha=.05)
    ax.set_xlim(10,50);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.suptitle('Receiver screening: uniform dim-sky scenario, not measured hardware',fontsize=12)
fig.savefig(ROOT.parent/'figures/receiver_requirements.png',dpi=190)

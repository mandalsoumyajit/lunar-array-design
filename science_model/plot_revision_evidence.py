"""Make the manuscript evidence figure from saved validation results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
fig, axes = plt.subplots(1,2,figsize=(10.5,3.65),layout='constrained')
for name, marker, label in [('response_recovery/vacuum_reference_lowband','o','Original database range'),
                            ('power_fix/controls','s','Local database + physical fields'),
                            ('power_fix/controls_fine','x','Corrected, finer mesh')]:
    rows = json.loads((ROOT/f'results/{name}/validation.json').read_text())['checks']
    rows = [r for r in rows if r.get('er',1)==1]
    axes[0].plot([r['frequency_mhz'] for r in rows],
                 [r.get('power_ratio',r.get('integrated_power_over_accepted')) for r in rows],
                 marker=marker, markersize=8, linestyle='none', label=label)
axes[0].axhline(1,color='#777777',ls='--',lw=1)
axes[0].set(xlabel='Frequency (MHz)',ylabel='Integrated field power / accepted power',
            title='(a) Isolated PEC dipole in vacuum',xlim=(0,30),ylim=(0,1.15))
axes[0].set_xticks([1,5,10,25]); axes[0].legend(frameon=False,loc='lower right',fontsize=8)
rows = json.loads((ROOT/'results/retrieval_corrected/stress_validation.json').read_text())['deterministic']
select = [next(r for r in rows if r['case']==c) for c in
          ('matched_40mK','foreground_fourth_order','foreground_ripple','inter_anchor_response')]
x = np.arange(4)
axes[1].errorbar(x,[r['fitted_amplitude_mk'] for r in select],
                 yerr=[r['sigma_mk'] for r in select],fmt='o',capsize=4,color='#0066a1')
axes[1].axhline(40,color='#777777',ls='--',lw=1,label='40 mK injection')
axes[1].set_xticks(x,['Matched','Smooth\nforeground','Foreground\nripple','Between\nanchors'])
axes[1].set(ylabel='Recovered amplitude (mK)',title='(b) Controlled retrieval stress tests',ylim=(-65,355))
axes[1].legend(frameon=False,loc='upper left')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
fig.savefig(ROOT.parent/'figures/revision_evidence.png',dpi=220)

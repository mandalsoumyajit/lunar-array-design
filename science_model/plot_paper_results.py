"""Paper figures from retained science/timing tables and checked station responses."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from response_bridge import load_physical_station_response
from receiver_requirements import circuit, FREQUENCIES

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/paper_evidence'
FIG=ROOT.parent/'figures'
inputs={}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    inputs[str(p.relative_to(ROOT.parent))]=sha(p)
    with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=230,bbox_inches='tight')
    plt.close(fig)
def panels(axes,titles):
    for ax,title in zip(axes,titles):
        ax.set_title(title,loc='left',fontsize=10)
        ax.grid(alpha=.18)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False})
blue,red,gold,green='#0066A1','#C23B33','#C88B16','#328457'
summary={}

r=rows(DATA/'ares_default_global21cm.csv')
f=np.array([float(x['frequency_mhz']) for x in r])
t=np.array([float(x['delta_tb_mk']) for x in r])
idx=np.argsort(f);f=f[idx];t=t[idx]
fig,axs=plt.subplots(1,2,figsize=(10,3.15),layout='constrained')
for ax,limit in zip(axs,[(5,205),(5,50)]):  # ARES reionization ends abruptly near 215 MHz
    ax.plot(f,t,color=blue,lw=2)
    ax.axhline(0,color='gray',lw=.8)
    ax.axvspan(14,46,color=green,alpha=.12,label='14-46 MHz priority')
    ax.set_xlim(*limit);ax.set_xlabel('Observed frequency (MHz)')
    sec=ax.secondary_xaxis('top',functions=(lambda x:1420.40575/np.maximum(x,1e-8)-1,lambda x:1420.40575/np.maximum(x+1,1e-8)))
    sec.set_xlabel('Redshift')
    sec.set_xticks([6,10,20,50,100] if limit[1]>50 else [30,50,100,200])
axs[0].set_ylabel('Global 21-cm brightness relative to CMB (mK)')
axs[0].set_ylim(-145,40)
axs[0].annotate('Cosmic Dawn absorption',xy=(67.58,-128.99),xytext=(93,-107),fontsize=8,arrowprops={'arrowstyle':'->','color':'gray'})
axs[0].annotate('Emission before reionization',xy=(117.79,15.62),xytext=(113,-40),fontsize=8,arrowprops={'arrowstyle':'->','color':'gray'})
low=(f>=5)&(f<=50)
axs[1].set_ylim(min(t[low])*1.18,4);axs[1].set_ylabel('Global brightness (mK)')
axs[1].legend(loc='lower right',frameon=False)
panels(axs,['(a) Illustrative ARES brightness spectrum','(b) Low-frequency collisional absorption'])
save(fig,'science_brightness')
summary['ares']={'global_minimum_mk':float(t.min()),'priority_minimum_mk':float(t[(f>=14)&(f<=46)].min()),'samples':len(f)}

zs=[];resp={}
for freq in FREQUENCIES:
    p=ROOT/f'results/power_fix/station/physical_lossy_f{freq}/response.npz'
    load_physical_station_response(p)
    inputs[str(p.relative_to(ROOT.parent))]=sha(p)
    with np.load(p) as d:
        z=d['z_ohm'];zs.append(z)
        L,_,_=circuit(z,freq*1e6,3,1e6)
        loaded=np.einsum('ab,nbc->nac',L,d['hoc_m'])
        a=d['angles_deg'];selected=np.isclose(a[:,1],0)
        resp[freq]=(a[selected,0],np.sqrt(np.sum(abs(loaded[selected,2,:])**2,axis=1)))
zs=np.array(zs);freqs=np.array(FREQUENCIES)
fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
axs[0].semilogy(freqs,abs(zs[:,0,0]),'o-',color=blue,label='LF self impedance')
axs[0].semilogy(freqs,abs(zs[:,2,2]),'s-',color=red,label='HF self impedance')
axs[0].set_ylabel(r'$|Z_{pp}|$ ($\Omega$)');axs[0].legend(frameon=False)
axs[1].plot(freqs,zs[:,0,2].real,'o-',color=blue,label=r'Re $Z_{13}$')
axs[1].plot(freqs,zs[:,0,2].imag,'s--',color=red,label=r'Im $Z_{13}$')
axs[1].axhline(0,color='gray',lw=.7);axs[1].set_ylabel(r'Mutual impedance ($\Omega$)');axs[1].legend(frameon=False)
for freq,c in zip([10,20,30,46],[blue,red,gold,green]):
    theta,h=resp[freq];axs[2].plot(theta,h,label=f'{freq} MHz',color=c)
axs[2].set_xlabel('Zenith angle (degrees)');axs[2].set_ylabel('Loaded effective-length norm (m)')
axs[2].legend(frameon=False,ncol=2,loc='upper right')
for ax in axs[:2]:ax.set_xlabel('Frequency (MHz)')
panels(axs,['(a) Different electrical scales','(b) Co-aligned LF/HF coupling','(c) HF receive cut, 3 pF inputs'])
save(fig,'corrected_station_response')
summary['station']={'maximum_sampled_mutual_impedance_ohm':float(abs(zs[:,0,2]).max()),'frequency_at_maximum_mhz':int(freqs[np.argmax(abs(zs[:,0,2]))]),'response_cut_phi_deg':0,'response_norm':'sqrt(sum over both transverse incident polarizations of |h_loaded[HF X]|^2))'}

fiber=rows(DATA/'fiber_cross_validation_metrics.csv')
fso=rows(DATA/'fso_link_budget_exact.csv')  # exact offset-Gaussian capture (fso_exact_capture.py)
fig,axs=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
nu=np.linspace(10,125,300)
limit=np.sqrt(-2*np.log(.999))/(2*np.pi*nu*1e6)*1e12
axs[0].semilogy(nu,limit,color=blue,label='0.1% coherence-loss limit')
axs[0].axhline(50,color=red,ls='--',label='50 ps test allocation')
axs[0].axvspan(14,46,color=green,alpha=.12)
axs[0].set_xlabel('Frequency (MHz)');axs[0].set_ylabel('Relative timing RMS (ps)');axs[0].legend(frameon=False,loc='upper right')
lengths=[.1,1,10];x=np.arange(3);width=.24
fiber_values={}
for i,(model,c,lab) in enumerate([('uncorrected',red,'Uncorrected'),('static-temperature correction',gold,'Static correction'),('lag-aware correction',green,'Lag-aware correction')]):
    vals=[float(next(r for r in fiber if r['sfp_reference'].startswith('1490') and r['model']==model and np.isclose(float(r['length_km']),l))['rms_ps']) for l in lengths]
    fiber_values[model]=vals;axs[1].bar(x+(i-1)*width,vals,width,color=c,label=lab)
axs[1].set_yscale('log');axs[1].set_ylim(2,600);axs[1].axhline(50,color='black',ls='--',lw=.8)
axs[1].set_xticks(x,['0.1','1','10']);axs[1].set_xlabel('Fiber length (km)');axs[1].set_ylabel('Held-out segment residual RMS (ps)')
axs[1].legend(frameon=False,fontsize=7,loc='upper left')
divs=[.1,.25,.5,1];closure=np.zeros((4,3))
for i,d in enumerate(divs):
    for j,l in enumerate(lengths):
        rr=next(r for r in fso if np.isclose(float(r['divergence_mrad']),d) and np.isclose(float(r['range_km']),l) and float(r['dust_loss_db'])==6 and float(r['background_penalty_db'])==0 and float(r['pointing_axis_rms_urad'])==100)
        closure[i,j]=float(rr['closure_probability'])
im=axs[2].imshow(closure,vmin=0,vmax=1,cmap='viridis',aspect='auto')
for i in range(4):
    for j in range(3):axs[2].text(j,i,f'{closure[i,j]:.2f}',ha='center',va='center',color='black' if closure[i,j]>.55 else 'white',fontsize=9)
axs[2].set_xticks(range(3),['0.1','1','10']);axs[2].set_yticks(range(4),['0.1','0.25','0.5','1'])
axs[2].set_xlabel('FSO range (km)');axs[2].set_ylabel('Divergence half-angle (mrad)')
panels(axs,['(a) Coherence sets a timing scale','(b) Thermal correction model','(c) Modeled link-closure probability'])
axs[2].grid(False)
save(fig,'timing_results')
summary['timing']={'fiber_rms_ps':fiber_values,'fso_closure_matrix':closure.tolist(),'fso_rows_divergence_mrad':divs,'fso_columns_range_km':lengths}

layouts={}
fig,axs=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
for name,c in [('compact',blue),('extended',red),('core_and_outer',green)]:
    rr=rows(ROOT/f'results/layout_{name}.csv')
    xy=np.array([[float(r['x_m']),float(r['y_m'])] for r in rr])
    delta=xy[:,None,:]-xy[None,:,:];bl=np.linalg.norm(delta,axis=2)[np.triu_indices(len(xy),1)]
    b_lambda=np.sort(bl/(299792458/20e6))
    layouts[name]={'nodes':len(xy),'baselines_below_10lambda':int((b_lambda<10).sum()),'maximum_baseline_km':float(bl.max()/1000)}
    if name=='compact':axs[0].scatter(xy[:,0]/1000,xy[:,1]/1000,s=8,color=c)
    else:axs[1].scatter(xy[:,0]/1000,xy[:,1]/1000,s=8,color=c,marker='.' if name=='extended' else '+',label=name.replace('_',' '))
    axs[2].step(b_lambda,np.arange(1,len(bl)+1),where='post',label=name.replace('_',' '),color=c)
for ax in axs[:2]:ax.set_aspect('equal');ax.set_xlabel('East (km)');ax.set_ylabel('North (km)')
axs[1].legend(frameon=False,fontsize=7)
axs[2].set_xscale('log');axs[2].set_yscale('log');axs[2].axvline(10,color='gray',ls='--');axs[2].set_xlabel(r'Baseline length at 20 MHz ($\lambda$)');axs[2].set_ylabel('Cumulative baseline count');axs[2].legend(frameon=False,fontsize=7)
panels(axs,['(a) Compact layout, 128 nodes','(b) Extended and mixed layouts','(c) Short-spacing vs. extent trade'])
save(fig,'array_layout_results')
summary['layouts']=layouts
assert [layouts[n]['baselines_below_10lambda'] for n in ['compact','extended','core_and_outer']]==[558,0,301]

retr=rows(ROOT/'results/retrieval_corrected/retrieval_profile_results.csv')
stress=rows(ROOT/'results/retrieval_corrected/stress_injections.csv')
fig,axs=plt.subplots(1,2,figsize=(10,3.25),layout='constrained')
scens=['no_regolith_modes','two_modes_free','three_modes_free','three_modes_ensemble_prior']
bias=[];sig=[]
for s in scens:
    sub=[r for r in retr if r['scenario']==s]
    bias.append(max(abs(float(r['bias_mk'])) for r in sub));sig.append(float(sub[0]['posterior_sigma_mk']))
x=np.arange(4)
axs[0].bar(x-.18,bias,.36,label='Maximum absolute bias',color=blue)
axs[0].bar(x+.18,sig,.36,label=r'Conditional $1\sigma$',color=gold)
axs[0].set_yscale('log');axs[0].set_ylim(1,1e5);axs[0].axhline(40,color='gray',ls=':',label='40 mK injection')
axs[0].set_xticks(x,['No modes','2 free','3 free','3 with prior']);axs[0].set_ylabel('Signal amplitude scale (mK)');axs[0].legend(frameon=False,fontsize=8,loc='upper right')
keys=['matched_40mK','foreground_fourth_order','foreground_ripple','inter_anchor_response']
ss=[next(r for r in stress if r['case']==s) for s in keys]
axs[1].errorbar(x,[float(r['fitted_amplitude_mk']) for r in ss],yerr=[float(r['sigma_mk']) for r in ss],fmt='o',color=blue,capsize=4)
axs[1].axhline(40,color='gray',ls='--',label='40 mK injection')
axs[1].set_xticks(x,['Matched','Smooth\nforeground','Foreground\nripple','Between\nEM anchors'])
axs[1].set_ylabel('Recovered amplitude (mK)');axs[1].legend(frameon=False)
panels(axs,['(a) Proxy-fit bias and uncertainty','(b) Model mismatch defeats nominal recovery'])
save(fig,'calibration_results')
summary['retrieval']={'scenarios':scens,'maximum_bias_mk':bias,'conditional_sigma_mk':sig}
payload={'inputs_sha256':inputs,'plot_script_sha256':sha(Path(__file__)),'summary':summary,
         'scope':'Base figure bundle. The final paper build replaces layout and receiver panels with decision studies and records a consolidated final manifest.',
         'figures_sha256':{n:sha(FIG/(n+'.png')) for n in ['science_brightness','corrected_station_response','timing_results','array_layout_results','calibration_results']}}
(DATA/'figure_validation.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))

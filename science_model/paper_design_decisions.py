"""Decision-oriented supplements to the paper, using only gated station responses.
Run with ADS Python from the paper root. No existing EM solver is modified.
"""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from receiver_requirements import FREQUENCIES,noise_model,metrics,voltage_limit,circuit
from response_bridge import load_physical_station_response
from measurement_model import sky_covariance,C
from recover_response import digest,ETA,write_csv
ROOT=Path(__file__).parent;OUT=ROOT/'results/design_decisions';OUT.mkdir(exist_ok=True)
FIG=ROOT.parent/'figures';inputs={};data={}
def record(p): inputs[str(p.relative_to(ROOT.parent))]=digest(p)
for f in FREQUENCIES:
 p=ROOT/f'results/power_fix/station/physical_lossy_f{f}/response.npz'
 load_physical_station_response(p);record(p)
 with np.load(p) as a:data[f]={k:a[k].copy() for k in a.files}
def model(f,c,e=.7,i=.1,g=250):
 d=data[f];return noise_model(d['z_ohm'],d['sky_resistance_ohm'],d['ground_resistance_ohm'],f*1e6,c,e,i,sky_scale=.3,ground_k=g)
priority=[f for f in FREQUENCIES if 14<=f<=46]
bound=[]
for cap in np.arange(.5,8.001,.1):
 vals=[voltage_limit(model(f,cap,e=0),[2,3])[0] for f in priority]
 k=int(np.argmin(vals));bound.append(dict(capacitance_pf=float(cap),maximum_voltage_noise_nv=min(vals),limiting_frequency_mhz=priority[k]))
boundary_errors=[]
for r in bound[::15]:
 cap=r['capacitance_pf'];en=r['maximum_voltage_noise_nv']
 if en>0:
  achieved=max(metrics(model(f,cap,en),[2,3])['worst_nonsky_over_sky'] for f in priority)
  boundary_errors.append(abs(achieved-.25));assert abs(achieved-.25)<1e-10
write_csv(OUT/'receiver_boundary.csv',bound)
sensitivity=[]
# Receiver-added criterion: ground excluded. Ground term reported separately.
for name,scale in [('nominal',1),('joint_plus_5_percent',1.05),('joint_plus_10_percent',1.1),('joint_plus_20_percent',1.2)]:
 vals=[metrics(model(f,3*scale,.7*scale,.1*scale),[2,3])['worst_nonsky_over_sky'] for f in priority]
 k=int(np.argmax(vals));sensitivity.append(dict(case=name,capacitance_pf=3*scale,en_nv=.7*scale,in_pa=.1*scale,ground_k='',worst_ratio=vals[k],limiting_frequency_mhz=priority[k]))
for ground in [200,250,260]:
 vals=[metrics(model(f,3,g=ground),[2,3]) for f in priority]
 g=[v['ground_over_sky'] for v in vals];t=[v['worst_total_over_sky'] for v in vals];k=int(np.argmax(g))
 sensitivity.append(dict(case=f'ground_term_{ground}K',capacitance_pf=3,en_nv=.7,in_pa=.1,ground_k=ground,worst_ratio=g[k],limiting_frequency_mhz=priority[k],total_with_receiver=max(t),ground_by_frequency=';'.join(f'{x:.3f}' for x in g)))
keys=list(dict.fromkeys(k for r in sensitivity for k in r));write_csv(OUT/'receiver_sensitivity.csv',[{k:r.get(k,'') for k in keys} for r in sensitivity])
# Rebuild existing receiver panels and replace the allowance plot by a band envelope.
source=ROOT/'results/receiver_requirements';comp=list(csv.DictReader((source/'comparison.csv').open()));thermal=list(csv.DictReader((source/'passive_temperature_and_correlation_screen.csv').open()));record(source/'comparison.csv');record(source/'passive_temperature_and_correlation_screen.csv')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8})
fig,ax=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained')
for case,label,col in [('reference','35 pF, 3 nV','#C23B33'),('capacitance_only','10 pF, 3 nV','#C88B16'),('prototype_target','3 pF, 0.7 nV, 0.1 pA','#0066A1')]:
 r=[x for x in comp if x['case']==case and x['band']=='HF' and x['sky_scale']=='0.3' and float(x['frequency_mhz'])>=10]
 ax[0].semilogy([float(x['frequency_mhz']) for x in r],[float(x['worst_nonsky_over_sky']) for x in r],'-o',ms=3,label=label,color=col)
r=[x for x in thermal if x['ground_temperature_k']=='250' and x['bias_and_series_temperature_c']=='125']
ax[0].semilogy([float(x['frequency_mhz']) for x in r],[float(x['largest_tested_nonsky_over_sky']) for x in r],'--',color='#0066A1',label='Target + passive/correlation stress')
ax[0].semilogy([float(x['frequency_mhz']) for x in r],[float(x['ground_only_over_sky']) for x in r],'-.',color='#555555',label='Ground emission, 250 K (separate term)')
ax[0].axhline(.25,color='gray',ls=':');ax[0].axvspan(14,46,color='#0066A1',alpha=.06);ax[0].set(xlabel='Frequency (MHz)',ylabel='Noise power / sky noise power',title='(a) Receiver-added noise');ax[0].legend(frameon=False)
x=[r['capacitance_pf'] for r in bound];y=[r['maximum_voltage_noise_nv'] for r in bound]
ax[1].fill_between(x,0,y,color='#328457',alpha=.18,label='Feasible at all nine priority samples');ax[1].plot(x,y,color='#328457');ax[1].plot(3,.7,'o',color='#0066A1',label='Chosen comparison point');ax[1].set(xlabel='Differential input capacitance (pF)',ylabel='Voltage noise (nV/sqrt(Hz))',title='(b) Sampled-band feasible region',ylim=(0,max(y)*1.08));ax[1].legend(frameon=False)
for a in ax:a.grid(alpha=.16)
fig.savefig(FIG/'receiver_requirements.png',dpi=230);plt.close(fig)
# End-to-end mean visibility from absolute loaded HF Jones matrices. Two identical
# stations, separate single unpolarized sources. Position perturbation is translation,
# not shape/height error: the station Jones remains fixed except for the load test.
f=46;d=data[f];pos=np.array([[0.,0,0],[100.,0,0]]);poserr=pos.copy();poserr[0,0]+=.01
L=circuit(d['z_ohm'],f*1e6,3,1e6)[0];Lp=circuit(d['z_ohm'],f*1e6,3.3,1e6)[0]
vis=[];phases=[];design=[]
for theta,phi in [(30,60),(60,0)]:
 ix=np.flatnonzero(np.all(d['angles_deg']==[theta,phi],axis=1))[0];t,p=np.deg2rad([theta,phi]);direction=np.array([[np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)]])
 J=(L@d['hoc_m'][ix])[2:]*np.sqrt(ETA*1e-26/2);Jp=(Lp@d['hoc_m'][ix])[2:]*np.sqrt(ETA*1e-26/2);jj=np.broadcast_to(J,(2,1,2,2)).copy();jp=jj.copy();jp[0,0]=Jp
 ref=sky_covariance(pos,direction,[[1000,0,0,0]],f*1e6,jj)[0,2]
 variants={'clock':(pos,jj,[50e-12,0]),'survey':(poserr,jj,[0,0]),'clock_and_survey':(poserr,jj,[50e-12,0]),'load_only':(pos,jp,[0,0]),'combined':(poserr,jp,[50e-12,0])}
 for case,(r,j,delay) in variants.items():
  v=sky_covariance(r,direction,[[1000,0,0,0]],f*1e6,j,delay)[0,2];ratio=v/ref
  vis.append(dict(theta_deg=theta,phi_deg=phi,case=case,nominal_visibility_abs_v2_hz=float(abs(ref)),amplitude_change_percent=float(100*(abs(ratio)-1)),phase_change_deg=float(np.rad2deg(np.angle(ratio)))))
  if case=='clock_and_survey':
   predicted=2*np.pi*f*1e6*(50e-12-.01*direction[0,0]/C);assert abs(np.angle(ratio)-predicted)<1e-10
   phases.append(np.angle(ratio)/(2*np.pi*f*1e6));design.append([1,-direction[0,0]/C])
 fitted=np.linalg.lstsq(np.array(design)*np.array([1e-12,.01]),phases,rcond=None)[0] if len(design)==2 else None
assert np.allclose(fitted,[50,1],rtol=1e-8,atol=1e-8)
write_csv(OUT/'integrated_visibility.csv',vis)
# Zenith snapshot dirty-beam cuts, equal cross-baseline weights, no autocorrelations.
# Euclidean MST: idealized routing scale for a tree of straight station-to-station segments (not a strict lower bound; Steiner junctions can shorten it).
layouts={};cost=[];cuts=[];beam_errors=[];angle=np.linspace(-2,2,4001);l=np.sin(np.deg2rad(angle));k=2*np.pi*20e6/C
for name in ['compact','extended','core_and_outer']:
 p=ROOT/f'results/layout_{name}.csv';record(p);r=list(csv.DictReader(p.open()));xy=np.array([[float(a['x_m']),float(a['y_m'])] for a in r]);layouts[name]=xy
 dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);tree=minimum_spanning_tree(dist);n=len(xy);assert tree.count_nonzero()==n-1
 af=np.exp(-1j*k*l[:,None]*xy[:,0]);beam=(abs(af.sum(axis=1))**2-n)/(n*(n-1));assert abs(beam[len(beam)//2]-1)<1e-12
 positive=beam[len(beam)//2:];idx=np.flatnonzero(positive<=.5)[0];u=angle[len(beam)//2:];half=np.interp(.5,positive[idx-1:idx+1][::-1],u[idx-1:idx+1][::-1]);cost.append(dict(layout=name,tree_length_km=float(tree.sum()/1000),ew_fwhm_arcmin=float(2*half*60)))
 for ix in [723,2000,2456]:
  dx=(xy[:,0,None]-xy[None,:,0])[np.triu_indices(n,1)]
  direct=float(np.mean(np.cos(k*l[ix]*dx)))
  beam_errors.append(abs(direct-beam[ix]));assert abs(direct-beam[ix])<1e-12
 cuts.append(beam)
write_csv(OUT/'layout_metrics.csv',cost)
fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
for name,col,beam in zip(layouts,['#0066A1','#C23B33','#328457'],cuts):
 xy=layouts[name];axs[0].scatter(xy[:,0]/1000,xy[:,1]/1000,s=6,label=name.replace('_',' '),color=col,alpha=.8);axs[1].plot(angle,beam,label=name.replace('_',' '),color=col)
axs[0].set(xlabel='East (km)',ylabel='North (km)',title='(a) Equal-count layouts');axs[0].set_aspect('equal');axs[1].legend(frameon=False,fontsize=7,loc='upper left')
axs[1].set(xlabel='East-west offset from zenith (deg)',ylabel='Normalized dirty-beam response',title='(b) 20 MHz snapshot beam cuts',xlim=(-1,1));axs[1].axhline(.5,ls=':',color='gray')
axs[2].bar(['Compact','Extended','Mixed'],[r['tree_length_km'] for r in cost],color=['#0066A1','#C23B33','#328457']);axs[2].set(ylabel='Spanning-tree length (km)',title='(c) Idealized routing length (MST)')
for a in axs:a.grid(alpha=.15)
fig.savefig(FIG/'array_layout_results.png',dpi=230);plt.close(fig)
for p in [Path(__file__),ROOT/'receiver_requirements.py',ROOT/'response_bridge.py',ROOT/'measurement_model.py',FIG/'receiver_requirements.png',FIG/'array_layout_results.png',*[OUT/n for n in ['receiver_boundary.csv','receiver_sensitivity.csv','integrated_visibility.csv','layout_metrics.csv']]]:record(p)
report=dict(checks=dict(max_boundary_ratio_error=max(boundary_errors),max_pair_sum_beam_error=max(beam_errors),connected_tree_edge_count=True,visibility_analytic_phase=True),status='Passed sampled-model checks; no hardware or continuous-band certification',receiver_sensitivity=sensitivity,priority_samples=priority,visibility=vis,geometry_clock_recovery=dict(delay_ps=float(fitted[0]),translation_m=float(.01*fitted[1])),layouts=cost,inputs_sha256=inputs)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['inputs_sha256','visibility']},indent=2))

"""Gate and publish the isolated-radiator high-band comparison; no normalization fit."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recover_response import ROOT,KB,digest,write_csv
from momentum_safe import verify_solve
out=ROOT/'results/high_band_validation'
def read(cell):
 p=out/f'cell_{cell:g}/metrics.csv'
 with p.open() as h:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(h)]
fine=read(.01);coarse=read(.025)
assert len(fine)==len(coarse)==12
lookup={(r['height_m'],r['frequency_mhz']):r for r in coarse}
for r in fine:
 c=lookup[r['height_m'],r['frequency_mhz']];z=complex(r['z_real_ohm'],r['z_imag_ohm']);zc=complex(c['z_real_ohm'],c['z_imag_ohm']);f=r['frequency_mhz']*1e6
 r['mesh_fraction_change']=r['sky_fraction']-c['sky_fraction'];r['mesh_z_relative_change']=abs(z-zc)/abs(z)
 # Hypothetical extension of target receiver; independent, uncorrelated e and i.
 sky=15000*(f/20e6)**-2.5;d=1/(1+z*(1e-6+2j*np.pi*f*3e-12))
 r['loaded_voltage_factor']=abs(d);r['assumed_sky_temperature_k']=sky
 # Receiver-added noise (amplifier + 300 K bias) relative to sky; regolith emission
 # (250 K subsurface brightness) is reported separately as ground_to_sky_ratio.
 r['excess_to_sky_ratio']=((.7e-9)**2/(4*KB*abs(d)**2)+abs(z)**2*((.1e-12)**2/(4*KB)+300/1e6))/(sky*r['sky_resistance_ohm'])
 r['ground_to_sky_ratio']=250*r['ground_resistance_ohm']/(sky*r['sky_resistance_ohm'])
for cell in [.01,.025]:
 for root in (out/f'cell_{cell:g}').glob('h*_f*'):
  verify_solve(next((root/'cases').iterdir()),[float(root.name.split('_f')[1])*1e6])
fields=json.loads((out/'absolute_field_audit.json').read_text());integration=json.loads((out/'integration_audit.json').read_text())
checks=dict(max_power_error=max(abs(r['power_error_fraction']) for r in fine+coarse),max_fine_power_error=max(abs(r['power_error_fraction']) for r in fine),max_sky_fraction_mesh_change=max(abs(r['mesh_fraction_change']) for r in fine),max_complex_impedance_mesh_change=max(r['mesh_z_relative_change'] for r in fine),max_upper_field_error=max(r['maximum_upper_field_relative_error'] for r in fields['checks']),max_quadrature_fraction_change=max(abs(r['quadrature_sky_fraction_change']) for r in integration),max_feed_gap_fraction_change=max(abs(r['finite_gap_sky_fraction_change']) for r in integration))
checks['height_order_preserved']=all([r['height_m'] for r in sorted([r for r in fine if r['frequency_mhz']==f],key=lambda r:r['sky_fraction'])]==[.1,.5,1.] for f in [50,75,100,125])
assert checks['max_power_error']<.005 and checks['max_sky_fraction_mesh_change']<.001 and checks['max_complex_impedance_mesh_change']<.01 and fields['passed'] and checks['height_order_preserved']
write_csv(out/'comparison.csv',fine)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
for height,color in zip([.1,.5,1],['#C23B33','#C88B16','#0066A1']):
 rows=sorted([r for r in fine if r['height_m']==height],key=lambda r:r['frequency_mhz']);f=[r['frequency_mhz'] for r in rows]
 for ax,key,scale in zip(axs,['sky_fraction','z_real_ohm','excess_to_sky_ratio'],[100,1,1]):
  ax.plot(f,[r[key]*scale for r in rows],'o-',color=color,label=f'{height:g} m',lw=1.5,ms=4)
 axs[2].plot(f,[r['ground_to_sky_ratio'] for r in rows],'--',color=color,lw=1,alpha=.8)
axs[0].set_ylabel('Power into sky / total radiated power (%)');axs[0].set_ylim(0,70);axs[0].legend(title='Dipole height',frameon=False,loc='upper left',fontsize=8,title_fontsize=8)
axs[1].set_ylabel('Input resistance (ohm)');axs[1].set_yscale('log')
axs[2].set_ylabel('Noise power / sky noise');axs[2].plot([],[],'k-',label='receiver-added');axs[2].plot([],[],'k--',lw=1,label='ground, 250 K');axs[2].legend(frameon=False,fontsize=7,loc='upper left');axs[2].set_yscale('log');axs[2].axhline(.25,color='gray',ls='--');axs[2].text(78,.21,'0.25 allocation',color='gray',fontsize=8)
for ax,title in zip(axs,['(a) Physical sky coupling','(b) Modal impedance variation','(c) Conditional receiver extension']):
 ax.set_title(title,loc='left',fontsize=10);ax.set_xlabel('Frequency (MHz)');ax.set_xticks([50,75,100,125]);ax.grid(alpha=.18)
figpath=ROOT.parent/'figures/high_band_validation.png';fig.savefig(figpath,dpi=230);plt.close(fig)
inputs=[out/'comparison.csv',out/'integration_audit.json',out/'absolute_field_audit.json',figpath,*[ROOT/n for n in ['validate_high_band.py','audit_high_band.py','audit_high_band_fields.py','summarize_high_band.py','physical_halfspace_fields.py','momentum_safe.py']]]
for cell in [.01,.025]:
 inputs+=[out/f'cell_{cell:g}/metrics.csv',out/f'cell_{cell:g}/validation.json']
 for root in (out/f'cell_{cell:g}').glob('h*_f*'):
  inputs += [root/'currents.npz',next((root/'cases').iterdir())/'solve_manifest.json']
payload=dict(passed=True,scope='Isolated PEC 6.001 m span, 1 mm width and gap; homogeneous lossless epsilon_r=4 half-space. Four frequency samples, three heights. Not the four-port station.',checks=checks,receiver='Conditional 3 pF || 1 Mohm, 0.7 nV/sqrtHz, 0.1 pA/sqrtHz uncorrelated; bias resistor 300 K, receiver-added criterion; lower-medium brightness 250 K reported separately, uniform sky 15000*(f/20MHz)^-2.5 K. No hardware or lossy layered-ground validation.',inputs_sha256={str(p.relative_to(ROOT.parent)):digest(p) for p in inputs})
(out/'validation.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))
for h in [.1,.5,1.]:print(h,'m receiver ratios',[round(r['excess_to_sky_ratio'],3) for r in sorted(fine,key=lambda r:r['frequency_mhz']) if r['height_m']==h],'ground',[round(r['ground_to_sky_ratio'],3) for r in sorted(fine,key=lambda r:r['frequency_mhz']) if r['height_m']==h])


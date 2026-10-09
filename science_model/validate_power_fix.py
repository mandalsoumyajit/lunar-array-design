"""Fresh-cache controlled solves and independent current-based radiation checks."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT, ETA, write_csv
from momentum_safe import solve, absolute_cut, verify_solve
from physical_halfspace_fields import current_quadrature, fields, integrate_power


def main():
    p=argparse.ArgumentParser();p.add_argument('--fine',action='store_true');args=p.parse_args()
    proposal=ROOT.parents[2]/'external/em_archive/ads_momentum_benchmark'
    out=ROOT/'results/power_fix'/('controls_fine' if args.fine else 'controls')
    rows=[]
    for er,height,fmhz in [(1,0,1),(1,0,5),(1,0,25),(4,0,5),(4,0,25),(4,6,5),(4,6,25)]:
        root=out/f'er{er}_h{height}_f{fmhz}';work=root/'cases'/f'er{er:04.1f}_h{height*1000:05d}mm'.replace('.','p')
        if not (work/'solve_manifest.json').exists():
            subprocess.run([sys.executable,str(proposal/'prepare_stage_a_case.py'),str(proposal/'output/simulation_input'),str(root),'--height-m',str(height),'--er',str(er),'--frequencies-mhz',str(fmhz),'--max-cell-m','0.025' if args.fine else '0.05'],check=True,capture_output=True)
            dsfile=solve(work,[fmhz*1e6],export_currents=True)
        else:dsfile=work/'Corrected.ds'
        verify_solve(work,[fmhz*1e6])
        with dataset.open_dataset_for_reading(dsfile) as ds:frame=ds['data'].to_dataframe()
        f=fmhz*1e6;s=frame.at[f,'S[1,1]'];z=50*(1+s)/(1-s);cur=1/(z+50)
        xyz,mom=current_quadrature(work,cur,order=2)
        # Known top-of-stack phase reference; no fitted phase or amplitude.
        errors=[]
        for phi in [0,90]:
            theta=np.arange(0,86,5)
            plan=f'VISUALIZATIONTYPE 0;\nPARAMETER FREQUENCY, UNITS GHz, PT {f/1e9:.17g};\nPARAMETER PHI, UNITS DEG, PT {phi};\nVAR THETA, UNITS DEG, '+', '.join(f'PT {t}' for t in theta)+';\nPORT 1,\nUNITS VOLT, UNITS DEG, AMPLITUDE 1 PHASE 0,\nUNITS OHM, UNITS DEG, AMPLITUDE 50 PHASE 0;\n'
            angles,ads=absolute_cut(work,plan,work/f'absolute_phi{phi}.ctf')
            calc=fields(xyz,mom,f,np.deg2rad(angles),np.deg2rad(phi),er,reference_z=height)
            errors.append(float(np.linalg.norm(calc-ads)/np.linalg.norm(ads)))
        if er==1:
            # Avoid an empty critical-angle interval in the vacuum limit.
            x,w=np.polynomial.legendre.leggauss(70);theta=np.arccos(x);phi=np.arange(90)*2*np.pi/90
            ff=fields(xyz,mom,f,theta[:,None],phi[None,:],er)
            powers=[float(np.sum(np.sum(abs(ff)**2,axis=-1)*w[:,None])*2*np.pi/90/(2*ETA))]
        else:powers=integrate_power(xyz,mom,f,er,ntheta=60,nphi=90)
        accepted=float(0.5*abs(cur)**2*z.real)
        row=dict(er=er,height_m=height,frequency_mhz=fmhz,real_z_ohm=float(z.real),power_ratio=sum(powers)/accepted,absolute_upper_field_relative_error=max(errors),accepted_w=accepted,physical_power_w=sum(powers))
        rows.append(row);print(row,flush=True)
    write_csv(out/'validation.csv',rows)
    payload=dict(passed=all(abs(r['power_ratio']-1)<.01 and r['absolute_upper_field_relative_error']<.005 for r in rows),checks=rows,
                 notes='Independent radiation from solved spatial currents; no fitted field scale or phase. Supported geometry is horizontal rectangular PEC strips.')
    (out/'validation.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':main()

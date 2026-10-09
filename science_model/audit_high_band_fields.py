"""Compare independent upper fields with native absolute fields on fresh solves."""
import json,numpy as np
from recover_response import ROOT
from momentum_safe import absolute_cut,verify_solve
from physical_halfspace_fields import fields
out=ROOT/'results/high_band_validation';rows=[]
for root in sorted((out/'cell_0.01').glob('h*_f*')):
 work=next((root/'cases').iterdir());d=np.load(root/'currents.npz');f=float(d['frequency_hz']);verify_solve(work,[f]);h=float(root.name.split('_')[0][1:]);z=d['z'].item();cur=1/(50+z)
 errors=[]
 for phi in [0,90]:
  plan=f'VISUALIZATIONTYPE 0;\nPARAMETER FREQUENCY, UNITS GHz, PT {f/1e9:.17g};\nPARAMETER PHI, UNITS DEG, PT {phi};\nVAR THETA, UNITS DEG, '+', '.join(f'PT {t}' for t in range(0,86,5))+';\nPORT 1,\nUNITS VOLT, UNITS DEG, AMPLITUDE 1 PHASE 0,\nUNITS OHM, UNITS DEG, AMPLITUDE 50 PHASE 0;\n'
  angles,ads=absolute_cut(work,plan,work/f'absolute_phi{phi}.ctf')
  calc=fields(d['xyz'],d['moments']*cur,f,np.deg2rad(angles),np.deg2rad(phi),4,reference_z=h)
  errors.append(float(np.linalg.norm(calc-ads)/np.linalg.norm(ads)))
 row=dict(height_m=h,frequency_mhz=f/1e6,maximum_upper_field_relative_error=max(errors));rows.append(row);print(row,flush=True)
(out/'absolute_field_audit.json').write_text(json.dumps(dict(passed=len(rows)==12 and all(r['maximum_upper_field_relative_error']<.005 for r in rows),checks=rows),indent=2)+'\n')

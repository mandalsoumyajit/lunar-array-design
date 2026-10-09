"""Audit integration, the finite-feed omission, and native absolute upper fields."""
import json
import numpy as np
from recover_response import ROOT,digest
from momentum_safe import absolute_cut
from physical_halfspace_fields import fields,integrate_power
out=ROOT/'results/high_band_validation';rows=[]
for h,fm in [(1,125),(.5,50)]:
 root=out/'cell_0.025'/f'h{h:g}_f{fm:g}';work=next((root/'cases').iterdir());d=np.load(root/'currents.npz');xyz=d['xyz'];mom=d['moments'];f=fm*1e6;z=d['z'].item()
 p=np.array(integrate_power(xyz,mom,f,4,72,96));base=json.loads((out/'cell_0.025/validation.json').read_text())['checks'];b=next(r for r in base if r['height_m']==h and r['frequency_mhz']==fm)
 gapxyz=np.array([[u*.001/(2*np.sqrt(3)),v*.001/(2*np.sqrt(3)),h] for u in [-1,1] for v in [-1,1]])
 gapmom=np.tile([-.001/4,0,0],(4,1));pg=np.array(integrate_power(np.vstack([xyz,gapxyz]),np.vstack([mom,gapmom]),f,4,48,64))
 row=dict(height_m=h,frequency_mhz=fm,quadrature_sky_fraction_change=float(p[0]/sum(p)-b['sky_fraction']),quadrature_power_relative_change=float(sum(p)/(b['upper_w']+b['lower_w'])-1),finite_gap_sky_fraction_change=float(pg[0]/sum(pg)-b['sky_fraction']),finite_gap_power_error=float(sum(pg)/(.5*z.real)-1))
 rows.append(row);print(row,flush=True)
(out/'integration_audit.json').write_text(json.dumps(rows,indent=2)+'\n')

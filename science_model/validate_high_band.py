"""Fresh-cache validation of the original isolated 6.001 m high-band radiator."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT,write_csv,digest
from momentum_safe import solve,verify_solve
from physical_halfspace_fields import current_quadrature,integrate_power

def main():
 p=argparse.ArgumentParser();p.add_argument('--cell',type=float,default=.025);p.add_argument('--frequencies',type=float,nargs='+',default=[50,75,100,125]);p.add_argument('--heights',type=float,nargs='+',default=[.1,.5,1]);a=p.parse_args()
 proposal=ROOT.parents[2]/'external/em_archive/ads_momentum_benchmark'
 out=ROOT/'results/high_band_validation'/f'cell_{a.cell:g}';out.mkdir(parents=True,exist_ok=True);rows=[]
 for h in a.heights:
  for fm in a.frequencies:
   root=out/f'h{h:g}_f{fm:g}';work=root/'cases'/f'er04p0_h{round(h*1000):05d}mm';f=fm*1e6
   if not (work/'solve_manifest.json').exists():
    subprocess.run([sys.executable,str(proposal/'prepare_stage_a_case.py'),str(proposal/'output/simulation_input'),str(root),'--height-m',str(h),'--er','4','--frequencies-mhz',str(fm),'--max-cell-m',str(a.cell)],check=True,capture_output=True)
    solve(work,[f],export_currents=True)
   verify_solve(work,[f])
   with dataset.open_dataset_for_reading(work/'Corrected.ds') as ds:frame=ds['data'].to_dataframe()
   s=frame.at[f,'S[1,1]'];z=50*(1+s)/(1-s)
   # Unit entering-port current. No field renormalization and no residual power.
   xyz,mom=current_quadrature(work,1,order=2)
   powers=integrate_power(xyz,mom,f,4,ntheta=48,nphi=64)
   accepted=.5*z.real;upper=powers[0];lower=sum(powers[1:])
   row=dict(height_m=h,frequency_mhz=fm,cell_m=a.cell,z_real_ohm=float(z.real),z_imag_ohm=float(z.imag),accepted_w=float(accepted),upper_w=upper,lower_w=lower,power_error_fraction=float((upper+lower)/accepted-1),sky_fraction=upper/(upper+lower),sky_resistance_ohm=2*upper,ground_resistance_ohm=2*lower)
   rows.append(row);print(json.dumps(row),flush=True)
   np.savez(root/'currents.npz',xyz=xyz,moments=mom,z=z,frequency_hz=f)
   write_csv(out/'metrics.csv',rows)
   (out/'validation.json').write_text(json.dumps(dict(complete=len(rows)==len(a.heights)*len(a.frequencies),power_gate=.005,passed=all(abs(r['power_error_fraction'])<.005 for r in rows),checks=rows,source_sha256={n:digest(Path(__file__).parent/n) for n in ['validate_high_band.py','physical_halfspace_fields.py','momentum_safe.py']}),indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()

"""Lossless one-port half-space controls for the station power audit."""
import json
import re
import subprocess
import sys
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT, ETA, parse_fields, antenna_report, write_csv
from investigate_power_balance import ENGINE, environment, export


def main():
    proposal=ROOT.parents[2]/'external/em_archive/ads_momentum_benchmark'
    out=ROOT/'results/power_investigation/halfspace'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for height in [0,6]:
        root=out/f'h{height}'
        subprocess.run([sys.executable,str(proposal/'prepare_stage_a_case.py'),str(proposal/'output/simulation_input'),str(root),'--height-m',str(height),'--er','4','--frequencies-mhz','5','25','--max-cell-m','0.05'],check=True)
        work=root/'cases'/f'er04p0_h{height*1000:05d}mm'
        (work/'momentum.cfg').write_text('MOMSDB_USE_QSFREQUENCY=0.0001\n')
        data=root/'data'; data.mkdir(exist_ok=True)
        result=subprocess.run([str(ENGINE),'-O','-3D','--objMode=MW','--dsName=Control','--dsDir='+str(data),'proj','proj'],cwd=work,env=environment(),capture_output=True,text=True,check=True)
        (root/'solve.log').write_text(result.stdout+result.stderr)
        with dataset.open_dataset_for_reading(data/'Control.ds') as ds:frame=ds['data'].to_dataframe()
        for f in frame.index:
            s=frame.at[f,'S[1,1]']; z=50*(1+s)/(1-s); current=1/(z+50); paccept=abs(current)**2*z.real/2
            _,_,report=export(work,root/'fields',f,50)
            angles,field=parse_fields(work/'proj.fff')
            ant=(work/'proj.ant').read_text()
            theta,phi=map(float,re.search(r'Angle of U_max (\S+) (\S+)',ant).groups())
            # Independent scale from the report's stated direction, interpolated on 1-degree grid.
            atphi=angles[:,1]==round(phi)
            vals=np.sum(abs(field[atphi])**2,axis=1)
            norm_at_u=np.interp(theta,angles[atphi,0],vals)
            w=np.sin(np.deg2rad(angles[:,0]))*np.deg2rad(1)**2
            medium=np.where(angles[:,0]<90,1,np.where(angles[:,0]>90,2,1.5))
            power=float(np.sum(np.sum(abs(field)**2,axis=1)*w*medium)*report['emax_v']**2/(2*ETA))
            rows.append(dict(height_m=height,frequency_mhz=f/1e6,z_real_ohm=float(z.real),accepted_w=float(paccept),reported_over_accepted=report['radiated_power_w']/paccept,integrated_using_documented_scale_over_accepted=power/paccept,normalized_field_power_at_report_max=float(norm_at_u),integrated_using_direction_scale_over_accepted=power/norm_at_u/paccept))
    write_csv(out/'comparison.csv',rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

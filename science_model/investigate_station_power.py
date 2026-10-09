"""Compare source units and hemispheric power using preserved station currents."""
import shutil
import subprocess
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT, ETA, parse_fields, antenna_report, write_csv
from investigate_power_balance import ENGINE, environment


def main():
    case='lf6m_hf0p5m_rot000_lfband'
    source=ROOT.parents[2]/'external/em_archive/ads_momentum_benchmark/output/crossed_station_pilot'
    out=ROOT/'results/power_investigation/station_sources'
    work=out/'work'; work.mkdir(parents=True,exist_ok=True)
    for file in (source/'cases'/case).iterdir():
        if file.is_file(): shutil.copy2(file,work/file.name)
    with dataset.open_dataset_for_reading(source/'data'/f'Crossed_{case}.ds') as ds:
        frame=ds['data'].to_dataframe()
    rows=[]
    for f in [5e6,25e6]:
        s=np.array([[frame.at[f,f'S[{a+1},{b+1}]'] for b in range(4)] for a in range(4)])
        z=50*np.linalg.solve((np.eye(4)-s).T,(np.eye(4)+s).T).T
        for port in [0,2]:
            for mode in ['all50','load_rad','load_deg']:
                load=50 if mode=='all50' else 1/(1e-6+2j*np.pi*f*35e-12)
                zs=np.full(4,load,complex); zs[port]=50
                vs=np.full(4,1e-12,complex); vs[port]=1
                current=np.linalg.solve(z+np.diag(zs),vs)
                paccept=float(np.vdot(current,z@current).real/2)
                plan=f'VISUALIZATIONTYPE 1;\nPARAMETER FREQUENCY, UNITS GHz,\nPT {f/1e9:.17g};\nPARAMETER PHI, UNITS DEG, PT 0;\nVAR THETA, UNITS DEG, PT 0;\n'
                for p in range(4):
                    unit='DEG' if mode=='load_deg' else 'RAD'
                    phase=np.angle(zs[p],deg=unit=='DEG')
                    plan+=f'PORT {p+1},\nUNITS VOLT, UNITS DEG, AMPLITUDE {vs[p].real:.17g} PHASE 0,\nUNITS OHM, UNITS {unit}, AMPLITUDE {abs(zs[p]):.17g} PHASE {phase:.17g};\n'
                (work/'proj.vpl').write_text(plan)
                result=subprocess.run([str(ENGINE),'-FF','proj','proj'],cwd=work,env=environment(),capture_output=True,text=True,check=True)
                stem=f'f{f/1e6:g}_p{port+1}_{mode}'
                (out/(stem+'.log')).write_text(result.stdout+result.stderr)
                for ext in ['vpl','fff','ant']:shutil.copy2(work/('proj.'+ext),out/(stem+'.'+ext))
                report=antenna_report(work/'proj.ant')
                angles,field=parse_fields(work/'proj.fff'); field*=report['emax_v']
                w=np.sin(np.deg2rad(angles[:,0]))*np.deg2rad(1)**2
                power=np.sum(abs(field)**2,axis=1)*w/(2*ETA)
                upper=float(np.sum(power[angles[:,0]<90])+0.5*np.sum(power[angles[:,0]==90]))
                lower=float(np.sum(power[angles[:,0]>90])+0.5*np.sum(power[angles[:,0]==90]))
                rows.append(dict(frequency_mhz=f/1e6,port=port+1,mode=mode,accepted_w=paccept,upper_vacuum_w=upper,lower_vacuum_formula_w=lower,report_w=report['radiated_power_w'],report_over_accepted=report['radiated_power_w']/paccept,upper_over_accepted=upper/paccept,full_vacuum_formula_over_accepted=(upper+lower)/paccept))
    write_csv(out/'comparison.csv',rows)
    for r in rows:print(r)


if __name__=='__main__':main()

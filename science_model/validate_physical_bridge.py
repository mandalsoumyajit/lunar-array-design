"""Check repaired response loading, the RIME bridge, and historical rejection."""
import json
from pathlib import Path
import numpy as np
from response_bridge import load_physical_station_response,load_station_response
from measurement_model import sky_covariance,visibility_covariance
from physical_station_solver import ROOT
from recover_response import digest

rows=[];receiver=[]
for f in [1,5,10,15,20,25,30,40,50]:
    directory=ROOT/f'physical_lossy_f{f}'
    angles,jones,noise,frequency=load_physical_station_response(directory/'response.npz')
    index=np.flatnonzero((angles[:,0]==30)&(angles[:,1]==60))[0]
    theta,phi=np.deg2rad(angles[index]);direction=[[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]]
    positions=np.array([[0,0,0],[40,0,0],[0,50,0],[60,70,0]],float)
    jj=np.broadcast_to(jones[index],(4,1,2,2))
    response=sky_covariance(positions,direction,[[1,0,0,0]],frequency,jones=jj)
    pairs=[(2*p,2*q) for p in range(4) for q in range(p+1,4)]
    forward=np.array([response[a,b] for a,b in pairs]);data=1000*forward
    recovered=float(np.vdot(forward,data).real/np.vdot(forward,forward).real)
    assert abs(recovered-1000)<1e-8
    total=1000*response
    for p in range(4):total[2*p:2*p+2,2*p:2*p+2]+=noise
    covariance,relation=visibility_covariance(total,pairs,1e5)
    assert np.linalg.eigvalsh(covariance).min()>-1e-12*np.linalg.norm(covariance)
    rows.append(dict(frequency_mhz=f,recovered_flux_jy=recovered,covariance_positive=True,
                     response_validation_sha256=digest(directory/'response_validation.json')))
    with np.load(directory/'response.npz') as d:z=d['z_ohm']
    for port in [0,2]:
        others=[i for i in range(4) if i!=port]
        inactive=1/(1e-6+2j*np.pi*frequency*35e-12)
        za=z[port,port]-z[port,others]@np.linalg.solve(z[np.ix_(others,others)]+inactive*np.eye(3),z[others,port])
        for capacitance in [10,35]:
            load=1/(1e-6+2j*np.pi*frequency*capacitance*1e-12)
            receiver.append(dict(frequency_mhz=f,port=port,capacitance_pf=capacitance,voltage_transfer_db=float(20*np.log10(abs(load/(za+load))))))
old=ROOT.parents[1]/'response_recovery/lf6m_hf0p5m_rot000_lfband/response_f020.00.npz'
try:load_station_response(old)
except ValueError:old_rejected=True
else:raise AssertionError('Historical invalid fields accepted')
try:load_physical_station_response(ROOT/'nominal_f25/response.npz')
except (KeyError,ValueError,RuntimeError):old_newgate_rejected=True
else:raise AssertionError('Historical candidate accepted by repaired-model gate')
report=dict(passed=True,checks=rows,historical_gate_rejects=old_rejected,
            new_gate_rejects_historical_candidate=old_newgate_rejected,receiver_screen=receiver,
            code_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('response_bridge.py')]},
            scope='Noiseless discrete-source interface and noise-covariance check; not an imaging/sensitivity forecast.')
(ROOT/'physical_bridge_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:val for key,val in report.items() if key!='receiver_screen'},indent=2))
print(json.dumps([r for r in receiver if r['frequency_mhz']>=40 and r['port']==2],indent=2))

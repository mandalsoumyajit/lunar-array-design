"""Export the corrected station response with direct plane-wave receive checks.

This is numerical reciprocity validation within the stated ideal feed/PEC model,
not an independent hardware or separate-solver benchmark.
"""
import argparse,json
from pathlib import Path
import numpy as np
from physical_station_solver import ROOT,prepare_vacuum,mesh_basis,reflection_operator,fourier_currents
from physical_halfspace_fields import fields
from validate_physical_station import verify_case
from recover_response import C,ETA,KB,digest


def incident_rhs(geometry,frequency,theta,phi,er,layer):
    k=2*np.pi*frequency/C;kt=k*np.sin(theta);q=k*np.cos(theta)
    ql=np.sqrt(layer*k*k-kt*kt+0j);qb=np.sqrt(er*k*k-kt*kt+0j)
    ql=np.where(ql.imag>0,ql.conj(),ql);qb=np.where(qb.imag>0,qb.conj(),qb)
    prop=np.exp(-2j*ql)
    def reflect(y0,y1,y2):
        r01=(y0-y1)/(y0+y1);r12=(y1-y2)/(y1+y2)
        return (r01+r12*prop)/(1+r01*r12*prop)
    rs=reflect(q,ql,qb);rp=reflect(1/q,layer/ql,er/qb)
    fx,fy=fourier_currents(geometry,kt*np.cos(phi),kt*np.sin(phi))
    h=geometry[-1]-1
    down=np.exp(1j*q[:,None]*h);up=np.exp(-1j*q[:,None]*h)
    bt=(np.cos(phi)[:,None]*fx+np.sin(phi)[:,None]*fy)*np.cos(theta)[:,None]*(down+rp[:,None]*up)
    bs=(-np.sin(phi)[:,None]*fx+np.cos(phi)[:,None]*fy)*(down+rs[:,None]*up)
    return np.stack([bt,bs],axis=-1)


def export(directory):
    record=verify_case(directory)
    if not record['lossy']:raise ValueError('Response export requires the nominal lossy-ground solution')
    with np.load(directory/'solution.npz') as d:
        f=float(d['frequency_hz']);z=d['z'];xyz=d['xyz'];mom=d['moments'];upper=d['upper'];below=d['below']
    work=prepare_vacuum(f,record['mesh_scale']);geometry=mesh_basis(work)
    er=4.4*(1-.01j);layer=3.4*(1-.005j);k=2*np.pi*f/C
    with np.load(work/'physical_vacuum_operator.npz') as d:a=d['matrix'].copy()
    a+=reflection_operator(geometry,f,er=er,layer_er=layer,order=record['order'],nphi=record['nphi'])
    # Direct receive solve: prescribe incoming E, leave terminal currents zero,
    # solve the internal currents, and recover the open-circuit terminal voltage.
    test=np.array([[0,0],[17,31],[43,79],[67,123],[83,257]],float)
    rhs=incident_rhs(geometry,f,*np.deg2rad(test).T,er,layer)
    b=rhs.transpose(1,0,2).reshape(a.shape[0],-1)
    received=b[:4]-a[:4,4:]@np.linalg.solve(a[4:,4:],b[4:])
    received=received.reshape(4,len(test),2).transpose(1,0,2)
    transmitted=fields(xyz,mom,f,*np.deg2rad(test).T,er=er,layer_er=layer,layer_depth=1).transpose(0,2,1)*4*np.pi/(1j*k*ETA)
    receive_error=float(np.linalg.norm(received-transmitted)/np.linalg.norm(received))
    load=1/(1e-6+2j*np.pi*f*35e-12)
    loading=np.linalg.solve((z+load*np.eye(4)).T,load*np.eye(4)).T
    loaded_operator=a.copy();loaded_operator[:4,:4]+=load*np.eye(4)
    loaded=load*np.linalg.solve(loaded_operator,b)[:4].reshape(4,len(test),2).transpose(1,0,2)
    predicted=np.einsum('pq,dqa->dpa',loading,transmitted)
    loaded_error=float(np.linalg.norm(loaded-predicted)/np.linalg.norm(loaded))
    theta,phi=np.meshgrid(np.arange(0,90,5),np.arange(0,360,5),indexing='ij')
    angles=np.column_stack([theta.ravel(),phi.ravel()])
    hoc=fields(xyz,mom,f,*np.deg2rad(angles).T,er=er,layer_er=layer,layer_depth=1).transpose(0,2,1)*4*np.pi/(1j*k*ETA)
    hload=np.einsum('pq,dqa->dpa',loading,hoc)
    # Reciprocity transposes transmit power matrices into receive covariance.
    sky=upper.T;ground=below.T
    ground_psd=4*KB*100*(loading@ground@loading.conj().T)
    parallel=np.linalg.inv(np.linalg.inv(z)+np.eye(4)/load)
    receiver=9e-18*np.eye(4)+4*KB*300/1e6*(parallel@parallel.conj().T)
    equilibrium=loading@(sky+ground)@loading.conj().T+parallel@parallel.conj().T/1e6
    expected=(parallel+parallel.conj().T)/2
    equilibrium_error=float(np.linalg.norm(equilibrium-expected)/np.linalg.norm(expected))
    passed=bool(max(receive_error,loaded_error,equilibrium_error)<.001 and np.linalg.eigvalsh(ground_psd).min()>0)
    np.savez_compressed(directory/'response.npz',frequency_hz=f,angles_deg=angles,z_ohm=z,loading=loading,
                        hoc_m=hoc,hload_m=hload,sky_resistance_ohm=sky,ground_resistance_ohm=ground,
                        ground_psd_v2_hz=ground_psd,receiver_psd_v2_hz=receiver)
    report=dict(passed_numerical_response_checks=passed,frequency_hz=f,receive_relative_error=receive_error,
                loaded_receive_relative_error=loaded_error,thermal_equilibrium_relative_error=equilibrium_error,
                power_validation_sha256=digest(directory/'validation.json'),response_sha256=digest(directory/'response.npz'),
                exporter_sha256=digest(Path(__file__)),
                scope='Numerically checked ideal PEC station with balanced impressed feeds and planar ground. Uniform ground 100 K; illustrative 1 Mohm/35 pF, 3 nV/sqrt(Hz) receiver. No measured hardware/sensitivity validation.')
    (directory/'response_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise RuntimeError(f'Receive validation failed: {report}')
    print(directory.name,receive_error,loaded_error,equilibrium_error,flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--frequency-mhz',type=float,default=25);p.add_argument('--all',action='store_true');args=p.parse_args()
    for f in ([1,5,10,15,20,25,30,40,50] if args.all else [args.frequency_mhz]):export(ROOT/f'physical_lossy_f{f:g}')

if __name__=='__main__':main()

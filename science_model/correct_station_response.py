"""Candidate station response using local-cache solves and physical currents.

The lossless control TESTS the full four-port radiation matrix; currently FAILS.
No station response from this investigation is validated for science. For the lossy
station only upper-vacuum radiation is exported; material noise uses the passive
network residual at a declared uniform temperature, not a lower far-field report.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT,C,ETA,KB,digest
from momentum_safe import solve,absolute_cut,verify_solve
from physical_halfspace_fields import current_quadrature,fields


def main():
    p=argparse.ArgumentParser();p.add_argument('--frequency-mhz',type=float,default=25)
    p.add_argument('--lossless',action='store_true');p.add_argument('--integration-order',type=int,default=36)
    p.add_argument('--vacuum',action='store_true');p.add_argument('--no-port-solver',action='store_true')
    args=p.parse_args();f=args.frequency_mhz*1e6
    case='lf6m_hf0p5m_rot000_'+('lfband' if f<=30e6 else 'highband')
    source=ROOT.parents[2]/'external/em_archive/ads_momentum_benchmark/output/crossed_station_pilot/cases'/case
    label=f'{"lossless" if args.lossless else "nominal"}_f{args.frequency_mhz:g}'
    if args.vacuum:label+='_vacuum'
    if args.no_port_solver:label+='_noportsolver'
    work=ROOT/'results/power_fix/station'/label;work.mkdir(parents=True,exist_ok=True)
    if not (work/'solve_manifest.json').exists():
        for name in ['proj_a','proj.pin','proj.prt','proj.ltd','proj.opt','proj.params','proj.plan','proj.toi']:
            shutil.copy2(source/name,work/name)
        (work/'source_manifest.json').write_text(json.dumps({name:digest(source/name) for name in ['proj_a','proj.pin','proj.prt','proj.ltd','proj.opt']},indent=2))
        (work/'proj.sti').write_text(f'PT {f/1e9:.17g},\n;\n')
        opt=(work/'proj.opt').read_text();old=int(re.search(r'no_cells_per_wavelength (\d+)',opt)[1])
        opt=re.sub(r'no_cells_per_wavelength \d+;',f'no_cells_per_wavelength {round(old*(30e6 if f<=30e6 else 50e6)/f)};',opt)
        if args.no_port_solver:opt=opt.replace('INCLUDEPORTSOLVER ON;','INCLUDEPORTSOLVER OFF;')
        (work/'proj.opt').write_text(opt)
        if args.lossless:
            ltd=(work/'proj.ltd').read_text();(work/'proj.ltd').write_text(re.sub(r'LOSSTANGENT=\S+','LOSSTANGENT=0',ltd))
        if args.vacuum:
            ltd=(work/'proj.ltd').read_text();(work/'proj.ltd').write_text(re.sub(r'(MATERIAL \S+ PERMITTIVITY=)\S+',r'\g<1>1',ltd))
        solve(work,[f],export_currents=True)
        # Large debug matrix is reproducible and not used by the radiation export.
        matrix=work/'ZMatrix.m'
        if matrix.exists():
            (work/'unused_matrix_sha256.txt').write_text(digest(matrix)+'\n')
            matrix.unlink()
    verify_solve(work,[f])
    with dataset.open_dataset_for_reading(work/'Corrected.ds') as ds:frame=ds['data'].to_dataframe()
    s=np.array([[frame.at[f,f'S[{a+1},{b+1}]'] for b in range(4)] for a in range(4)])
    z=50*np.linalg.solve((np.eye(4)-s).T,(np.eye(4)+s).T).T
    resistance=(z+z.conj().T)/2
    currentfile=work/'current_quadrature_v3.npz'
    current_manifest={p.name:digest(p) for p in [work/'proj.msh',work/'BasisSections.log',*work.glob('Imatrix.m.*'),Path(__file__).with_name('physical_halfspace_fields.py')]}
    cache_manifest=work/'current_quadrature_v3.json'
    if currentfile.exists() and cache_manifest.exists() and json.loads(cache_manifest.read_text())==current_manifest:
        with np.load(currentfile) as dat:xyz=dat['xyz'];mom=dat['moments']
    else:
        moments=[]
        for port in range(4):
            xyz,j=current_quadrature(work,np.eye(4)[:,port],order=2);moments.append(j)
        # Momentum's mesh z=0 is the bottom interface of the finite regolith
        # layer; the physical surface is at z=1 m in this specific station deck.
        xyz[:,2]-=1
        if not np.all(np.isclose(xyz[:,2],.5)|np.isclose(xyz[:,2],6)):
            raise RuntimeError('Unexpected station height coordinates')
        mom=np.stack(moments,axis=-1);np.savez_compressed(currentfile,xyz=xyz,moments=mom)
        cache_manifest.write_text(json.dumps(current_manifest,indent=2)+'\n')
    print(label,'current quadrature ready',flush=True)
    er=4.4 if args.lossless else 4.4*(1-.01j)
    layer=3.4 if args.lossless else 3.4*(1-.005j)
    if args.vacuum:er=layer=1.
    def response(theta,phi):
        return fields(xyz,mom,f,theta,phi,er=er,reference_z=6,layer_er=layer,layer_depth=1)
    load=1/(1e-6+2j*np.pi*f*35e-12)
    loading=np.linalg.solve((z+load*np.eye(4)).T,load*np.eye(4)).T
    errors=[]
    for port in range(4):
        zs=np.full(4,load,complex);zs[port]=50
        vs=np.full(4,1e-12,complex);vs[port]=1
        currents=np.linalg.solve(z+np.diag(zs),vs)
        theta=np.arange(0,86,5);phi=37
        plan=f'VISUALIZATIONTYPE 0;\nPARAMETER FREQUENCY, UNITS GHz, PT {f/1e9:.17g};\nPARAMETER PHI, UNITS DEG, PT {phi};\nVAR THETA, UNITS DEG, '+', '.join(f'PT {t}' for t in theta)+';\n'
        for n in range(4):
            plan+=f'PORT {n+1},\nUNITS VOLT, UNITS DEG, AMPLITUDE {vs[n].real:.17g} PHASE 0,\nUNITS OHM, UNITS DEG, AMPLITUDE {abs(zs[n]):.17g} PHASE {np.angle(zs[n],deg=True):.17g};\n'
        angles,ads=absolute_cut(work,plan,work/f'absolute_p{port+1}.ctf')
        calc=response(np.deg2rad(angles),np.deg2rad(phi))@currents
        errors.append(float(np.linalg.norm(calc-ads)/np.linalg.norm(ads)))
    print(label,'absolute-field errors',errors,flush=True)
    # Gauss quadrature excludes the exact horizon and splits lower critical angles.
    nt=args.integration_order;np_=2*nt
    bounds=[0,np.pi/2]
    if args.lossless:
        bounds+=sorted([np.pi-np.arcsin(1/np.sqrt(er)),np.pi-np.arcsin(np.sqrt(layer/er))])+[np.pi]
    bounds=sorted(set(bounds))
    matrices=[];upper_data=None
    for low,high in zip(bounds[:-1],bounds[1:]):
        x,w=np.polynomial.legendre.leggauss(nt);theta=(x+1)*(high-low)/2+low
        w=w*(high-low)/2*np.sin(theta);phi=np.arange(np_)*2*np.pi/np_
        g=response(theta[:,None],phi[None,:]);weight=np.broadcast_to(w[:,None]*2*np.pi/np_,g.shape[:2])
        matrix=np.einsum('abip,abiq,ab->pq',g.conj(),g,weight)/ETA*(1 if high<=np.pi/2 else np.sqrt(er))
        matrices.append(matrix)
        print(label,'integrated interval',low,high,flush=True)
        if high<=np.pi/2:
            angles=np.stack(np.broadcast_arrays(theta[:,None],phi[None,:]),axis=-1).reshape(-1,2)
            upper_data=(np.rad2deg(angles),g.reshape(-1,2,4),weight.ravel())
    sky=matrices[0].T;residual=resistance-sky
    checks=dict(frequency_mhz=args.frequency_mhz,lossless=args.lossless,vacuum=args.vacuum,
        maximum_absolute_upper_field_relative_error=max(errors),upper_field_errors=errors,
        minimum_network_resistance_eigenvalue=float(np.linalg.eigvalsh(resistance).min()),
        minimum_non_sky_resistance_eigenvalue=float(np.linalg.eigvalsh(residual).min()),
        maximum_s_singular_value=float(np.linalg.svd(s,compute_uv=False).max()),integration_order=nt,
        network_reciprocity_relative=float(np.linalg.norm(z-z.T)/np.linalg.norm(z)))
    if args.lossless:
        total=sum(matrices)
        np.savez_compressed(work/'power_matrices.npz',network=resistance,sky_transmit=matrices[0],total=total,components=np.array(matrices))
        print('power diagonal ratios',np.diag(total).real/np.diag(resistance).real,flush=True)
        vals,vec=np.linalg.eigh(resistance);rinv=(vec/np.sqrt(vals))@vec.conj().T
        relative=rinv@total@rinv
        checks['radiation_to_network_generalized_eigenvalues']=np.linalg.eigvalsh(relative).tolist()
        checks['full_matrix_power_error']=float(np.linalg.norm(relative-np.eye(4),2))
    local_pass=max(errors)<.005 and checks['minimum_non_sky_resistance_eigenvalue']>0
    checks['passed_local_checks']=local_pass
    checks['passed']=local_pass and args.lossless and checks['full_matrix_power_error']<.01
    checks['validated_for_science']=False
    checks['validation_status']='diagnostic_only: full station power closure and independent receive validation remain required'
    aa,g,w=upper_data;hoc=4*np.pi/(1j*(2*np.pi*f/C)*ETA)*g.transpose(0,2,1)
    hload=np.einsum('pq,dqa->dpa',loading,hoc)
    zp=np.linalg.solve(np.linalg.inv(z)+np.eye(4)/load,np.eye(4))
    tsky=50000*(f/20e6)**-2.5
    sky_psd=4*KB*tsky*(loading@sky@loading.conj().T)
    loss_psd=4*KB*100*(loading@residual@loading.conj().T)
    rec_psd=(3e-9)**2*np.eye(4)+4*KB*300/1e6*(zp@zp.conj().T)
    np.savez_compressed(work/'response.npz',frequency_hz=f,angles_deg=aa,solid_angle_weights=w,
        z_ohm=z,loading=loading,hoc_m=hoc,hload_m=hload,sky_resistance_ohm=sky,
        residual_resistance_ohm=residual,sky_psd_v2_hz=sky_psd,loss_psd_v2_hz=loss_psd,
        receiver_psd_v2_hz=rec_psd)
    checks['response_sha256']=digest(work/'response.npz')
    checks['code_sha256']={name:digest(Path(__file__).with_name(name)) for name in
        ['correct_station_response.py','physical_halfspace_fields.py','momentum_safe.py']}
    checks['current_export_sha256']=current_manifest
    checks['scope']='Nominal rectangular PEC station only; uniform 100 K non-sky bath and illustrative receiver noise; no measured sensitivity or independent receive solve.'
    (work/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()

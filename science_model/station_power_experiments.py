"""Physically equivalent vacuum experiments to isolate station closure failure."""
from pathlib import Path
import re,shutil,json,argparse
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from momentum_safe import solve,verify_solve
from physical_halfspace_fields import current_quadrature,fields
from recover_response import ETA

p=argparse.ArgumentParser();p.add_argument('variant',choices=['unreduced','minimal_stack','minimal_smallgap','smallgap_freespace','split_stack','air_only','one_bottom_layer','shift_bottom','gf_diagnostics','freespace','source_switch_off','network_qsfloor','qs_green_off','multifrequency','configured','regularized']);p.add_argument('--depth',type=float,default=1.5);p.add_argument('--layered',action='store_true');p.add_argument('--setting',action='append',default=[]);p.add_argument('--tag',default='');p.add_argument('--epsilon-offset',type=float,default=1e-6);args=p.parse_args()
if args.layered and args.variant not in ['gf_diagnostics','freespace','configured','regularized']:raise ValueError('Unsupported layered experiment')
if args.variant=='configured' and (not re.fullmatch('[a-z0-9_]+',args.tag) or not args.setting):raise ValueError('Named settings required')
if any(not re.fullmatch(r'MOM(?:3D|SDB)_[A-Z0-9_]+=[A-Za-z0-9.+-]+',s) for s in args.setting):raise ValueError('Invalid setting')
root=Path(__file__).parent/'results/power_fix/station';source=root/('lossless_f25' if args.layered else 'lossless_f25_vacuum_noportsolver')
work=root/('kernel_'+args.variant+(f'_{args.depth:g}' if args.variant=='shift_bottom' else '')+(f'_{args.epsilon_offset:g}' if args.variant=='regularized' else '')+('_'+args.tag if args.tag else '')+('_layered' if args.layered else ''));work.mkdir(exist_ok=True)
if not (work/'solve_manifest.json').exists():
    for name in ['proj_a','proj.pin','proj.prt','proj.ltd','proj.opt','proj.params','proj.plan','proj.toi','proj.sti']:
        shutil.copy2(source/name,work/name)
    path=work/'proj.opt';path.write_text(path.read_text().replace('INCLUDEPORTSOLVER ON;','INCLUDEPORTSOLVER OFF;'))
    if args.variant=='regularized':
        if not 0<args.epsilon_offset<.01:raise ValueError('Small positive regularization required')
        path=work/'proj.ltd';text=path.read_text()
        text=text.replace('END_MATERIAL',f'  MATERIAL AIR_GAP PERMITTIVITY={1+args.epsilon_offset:.15g} LOSSTANGENT=0 PERMEABILITY=1\nEND_MATERIAL')
        path.write_text(text.replace('LAYER Name=LF_to_HF_Vacuum HEIGHT=5.5 MATERIAL=AIR','LAYER Name=LF_to_HF_Vacuum HEIGHT=5.5 MATERIAL=AIR_GAP'))
    if args.variant=='unreduced':
        path=work/'proj.opt';path.write_text(path.read_text().replace('mesh_reduction ON;','mesh_reduction OFF;'))
    elif args.variant in ['minimal_stack','minimal_smallgap','smallgap_freespace']:
        if args.variant!='smallgap_freespace':
            path=work/'proj.ltd';path.write_text(re.sub(r'  LAYER Name=HF_to_Surface_Vacuum.*?END_STACK','  BOTTOM OPEN MATERIAL=AIR\nEND_STACK',path.read_text(),flags=re.S))
        if args.variant in ['minimal_smallgap','smallgap_freespace']:
            path=work/'proj_a'
            def shrink_gap(match):
                line=match[0];coords=np.array([[float(x),float(y)] for x,y in re.findall(r'(-?\d+\.\d+),(-?\d+\.\d+)',line)])
                axis=np.argmax(np.ptp(coords,axis=0));shift=np.sign(coords[:,axis].mean())*np.min(abs(coords[:,axis]))*.75
                iterator=iter(coords-np.eye(2)[axis]*shift)
                return re.sub(r'-?\d+\.\d+,-?\d+\.\d+',lambda m:'{:.10f},{:.10f}'.format(*next(iterator)),line)
            path.write_text(re.sub(r'^ADD P[^\n]+',shrink_gap,path.read_text(),flags=re.M))
            path=work/'proj.pin';path.write_text(re.sub(r'<([xy])>([^<]+)</\1>',lambda m:f'<{m[1]}>{float(m[2])*.25:.15g}</{m[1]}>',path.read_text()))
    elif args.variant=='split_stack':
        path=work/'proj.ltd'
        layers='\n'.join(f'  LAYER Name=Vacuum{i} HEIGHT=0.5 MATERIAL=AIR\n  INTERFACE Name=VacuumBoundary{i}' for i in range(10))+'\n  LAYER Name=Vacuum10 HEIGHT=0.5 MATERIAL=AIR'
        path.write_text(path.read_text().replace('  LAYER Name=LF_to_HF_Vacuum HEIGHT=5.5 MATERIAL=AIR',layers))
    elif args.variant=='air_only':
        path=work/'proj.ltd';path.write_text(path.read_text().replace('MATERIAL=REGOLITH_UPPER','MATERIAL=AIR').replace('MATERIAL=REGOLITH_LOWER','MATERIAL=AIR'))
    elif args.variant in ['one_bottom_layer','shift_bottom']:
        path=work/'proj.ltd';path.write_text(re.sub(r'  LAYER Name=HF_to_Surface_Vacuum.*?END_STACK',f'  LAYER Name=BelowHF HEIGHT={args.depth:g} MATERIAL=AIR\n  INTERFACE Name=BottomReference\n  BOTTOM OPEN MATERIAL=AIR\nEND_STACK',path.read_text(),flags=re.S))
    # These diagnostic keys are discovered in the installed executable. Their
    # supported semantics are unconfirmed; never promote them without controls.
    extra={'gf_diagnostics':'MOM3D_WRITE_GREENFUNCTIONS=TRUE\nMOM3D_WRITE_GFTOMATLAB=TRUE\n',
           'freespace':'MOM3D_USE_GFFREESPACE=TRUE\n',
           'smallgap_freespace':'MOM3D_USE_GFFREESPACE=TRUE\n',
           'network_qsfloor':'MOM3D_USE_QSFREQUENCY=0.0025\nMOM3D_WRITE_GREENFUNCTIONS=TRUE\nMOM3D_WRITE_GFTOMATLAB=TRUE\n',
           'qs_green_off':'MOM3D_USE_QSGREENFUNCTIONS=FALSE\nMOM3D_WRITE_GREENFUNCTIONS=TRUE\nMOM3D_WRITE_GFTOMATLAB=TRUE\n',
           'source_switch_off':'MOM3D_USE_SOURCESWITCHALGORITHM=FALSE\n'}.get(args.variant,'')
    if args.variant=='configured':extra='\n'.join(args.setting)+'\n'
    if extra:
        import momentum_safe
        original_configure=momentum_safe.configure
        def diagnostic_configure(*args,**kwargs):
            original_configure(*args,**kwargs)
            path=work/'momentum.cfg';path.write_text(path.read_text()+extra)
        momentum_safe.configure=diagnostic_configure
    frequencies=[5e6,25e6] if args.variant=='multifrequency' else [25e6]
    if args.variant=='multifrequency':(work/'proj.sti').write_text('PT 0.005,\nPT 0.025,\n;\n')
    solve(work,frequencies,export_currents=True)
verify_solve(work,[25e6])
with dataset.open_dataset_for_reading(work/'Corrected.ds') as ds:frame=ds['data'].to_dataframe()
s=np.array([[frame.at[25e6,f'S[{i+1},{j+1}]'] for j in range(4)] for i in range(4)])
z=50*np.linalg.solve((np.eye(4)-s).T,(np.eye(4)+s).T).T
mom=[]
for i in range(4):
    xyz,j=current_quadrature(work,np.eye(4)[:,i],2);mom.append(j)
mom=np.stack(mom,axis=-1)
if args.layered:
    xyz[:,2]-=1
    bounds=[0,np.pi/2,np.pi-np.arcsin(1/np.sqrt(4.4)),np.pi-np.arcsin(np.sqrt(3.4/4.4)),np.pi];bounds=sorted(bounds)
    total=np.zeros((4,4),complex)
    for low,high in zip(bounds[:-1],bounds[1:]):
        x,w=np.polynomial.legendre.leggauss(36);theta=(x+1)*(high-low)/2+low;w=w*(high-low)/2*np.sin(theta);phi=np.arange(72)*2*np.pi/72
        g=fields(xyz,mom,25e6,theta[:,None],phi[None,:],er=4.4,layer_er=3.4,layer_depth=1)
        total+=np.einsum('abip,abiq,a->pq',g.conj(),g,w)*2*np.pi/72/ETA*(1 if high<=np.pi/2 else np.sqrt(4.4))
else:
    x,w=np.polynomial.legendre.leggauss(48);theta=np.arccos(x);phi=np.arange(96)*2*np.pi/96
    g=fields(xyz,mom,25e6,theta[:,None],phi[None,:],er=1)
    total=np.einsum('abip,abiq,a->pq',g.conj(),g,w)*2*np.pi/96/ETA
r=(z+z.conj().T)/2;v,u=np.linalg.eigh(r);ri=(u/np.sqrt(v))@u.conj().T
result=dict(variant=args.variant,layered=args.layered,z_real=z.real.tolist(),z_imag=z.imag.tolist(),
    generalized_eigenvalues=np.linalg.eigvalsh(ri@total@ri).tolist(),height_planes=np.unique(xyz[:,2]).tolist())
(work/'experiment_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""Fresh dimensionally equivalent solve; diagnostic, not a fitted correction."""
from pathlib import Path
import re,shutil,json,argparse
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from momentum_safe import solve
from physical_halfspace_fields import current_quadrature,fields
from recover_response import ETA,digest

p=argparse.ArgumentParser();p.add_argument('--scale',type=float,default=1000);p.add_argument('--source',default='lossless_f25_vacuum_noportsolver');a=p.parse_args()
root=Path(__file__).parent/'results/power_fix/station';source=root/a.source
work=root/(a.source+f'_scale{a.scale:g}');work.mkdir(exist_ok=True)
f=25e6*a.scale
if not (work/'solve_manifest.json').exists():
    for name in ['proj_a','proj.pin','proj.prt','proj.ltd','proj.opt','proj.params','proj.plan','proj.toi']:
        shutil.copy2(source/name,work/name)
    path=work/'proj_a';text=path.read_text()
    text=re.sub(r'(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)',lambda m:f'{float(m[1])/a.scale:.15g},{float(m[2])/a.scale:.15g}',text)
    # UNITS precision is a grid declaration, not a coordinate.
    text=re.sub(r'UNITS MM,[^;]+;', 'UNITS MM,1000000;',text)
    path.write_text(text)
    path=work/'proj.pin';path.write_text(re.sub(r'<([xy])>([^<]+)</\1>',lambda m:f'<{m[1]}>{float(m[2])/a.scale:.15g}</{m[1]}>',path.read_text()))
    path=work/'proj.ltd';path.write_text(re.sub(r'HEIGHT=(\S+)',lambda m:f'HEIGHT={float(m[1])/a.scale:.15g}',path.read_text()))
    (work/'proj.sti').write_text(f'PT {f/1e9:.15g},\n;\n')
    solve(work,[f],export_currents=True)
with dataset.open_dataset_for_reading(work/'Corrected.ds') as ds:frame=ds['data'].to_dataframe()
s=np.array([[frame.at[f,f'S[{i+1},{j+1}]'] for j in range(4)] for i in range(4)])
z=50*np.linalg.solve((np.eye(4)-s).T,(np.eye(4)+s).T).T
mom=[]
for i in range(4):
    xyz,j=current_quadrature(work,np.eye(4)[:,i],2);mom.append(j)
xyz*=a.scale;xyz[:,2]-=1;mom=np.stack(mom,axis=-1)*a.scale
vacuum='vacuum' in a.source;er=1 if vacuum else 4.4;layer=1 if vacuum else 3.4
bounds=sorted(set([0,np.pi/2,np.pi-np.arcsin(1/np.sqrt(er)),np.pi-np.arcsin(np.sqrt(layer/er)),np.pi]))
total=np.zeros((4,4),complex)
for low,high in zip(bounds[:-1],bounds[1:]):
    x,w=np.polynomial.legendre.leggauss(36);t=(x+1)*(high-low)/2+low;w=w*(high-low)/2*np.sin(t);ph=np.arange(72)*2*np.pi/72
    g=fields(xyz,mom,25e6,t[:,None],ph[None,:],er=er,layer_er=layer,layer_depth=1)
    total+=np.einsum('abip,abiq,a->pq',g.conj(),g,w)*2*np.pi/72/ETA*(1 if high<=np.pi/2 else np.sqrt(er))
r=(z+z.conj().T)/2;v,u=np.linalg.eigh(r);ri=(u/np.sqrt(v))@u.conj().T
result=dict(scale=a.scale,source=a.source,generalized_eigenvalues=np.linalg.eigvalsh(ri@total@ri).tolist(),z_real=z.real.tolist(),z_imag=z.imag.tolist(),total_real=total.real.tolist(),total_imag=total.imag.tolist())
(work/'scale_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

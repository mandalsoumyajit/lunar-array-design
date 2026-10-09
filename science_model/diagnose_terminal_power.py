"""Compare exported MoM resistance with radiation of individual basis functions."""
from pathlib import Path
import re,json
import numpy as np
from physical_halfspace_fields import current_quadrature,fields
from recover_response import ETA

work=Path(__file__).parent/'results/power_fix/station/lossless_f25_vacuum_noportsolver'
text=(work/'ZMatrix.m').read_text()
n=int(re.search(r'Rows: (\d+)',text)[1])
z=np.zeros((n,n),complex)
for i,j,a,b in re.findall(r'Z\((\d+), (\d+)\) = ([^ ]+) \+ j\* ([^ ;]+)',text):
    z[int(i)-1,int(j)-1]=complex(float(a),float(b))
z+=np.triu(z,1).T
indices=[0,1,2,3,4,5,6,7,8,22,23]
frames={int(i):[list(map(float,a.split())),list(map(float,b.split()))] for i,a,b in
    re.findall(r'b\[(\d+)\].*?frame: ([^,]+), ([^\n]+)',(work/'Basis.m').read_text())}
if not (frames[9][0][2]==7 and frames[23][0][2]==1.5 and frames[24][0][2]==1.5):
    raise RuntimeError('Basis numbering changed; select the recorded height planes again')
mom=[]
for i in indices:
    b=np.zeros(n);b[i]=1
    xyz,m=current_quadrature(work,np.zeros(4),order=2,basis_coefficients=b)
    mom.append(m)
x,w=np.polynomial.legendre.leggauss(32);theta=np.arccos(x);phi=np.arange(64)*2*np.pi/64
g=fields(xyz,np.stack(mom,axis=-1),25e6,theta[:,None],phi[None,:],er=1)
r=np.einsum('abip,abiq,a->pq',g.conj(),g,w)*2*np.pi/64/ETA
matrix=z[np.ix_(indices,indices)].real
np.set_printoptions(precision=5,suppress=True,linewidth=180)
print('Radiation resistance',r.real)
print('MoM resistance minus 50-ohm terminal loads',matrix-np.diag([50]*8+[0]*3))
print('Difference',matrix-np.diag([50]*8+[0]*3)-r.real)
(work/'terminal_basis_diagnostic.json').write_text(json.dumps(dict(basis_indices_1based=[i+1 for i in indices],basis_frames={i+1:frames[i+1] for i in indices},radiation=r.real.tolist(),mom_without_terminal_load= (matrix-np.diag([50]*8+[0]*3)).tolist()),indent=2)+'\n')

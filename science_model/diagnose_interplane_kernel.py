"""Independent vacuum EFIE interaction integral for nonterminal basis functions."""
from pathlib import Path
import re,json
import numpy as np
from physical_halfspace_fields import current_quadrature
from recover_response import C,ETA

work=Path(__file__).parent/'results/power_fix/station/lossless_f25_vacuum_noportsolver'
text=(work/'BasisSections.log').read_text()
mesh=(work/'proj.msh').read_text()
cells=[int(i) for i in re.findall(r'c (\d+):',mesh)]
def basis(index):
    b=np.zeros(848);b[index-1]=1
    xyz,mom=current_quadrature(work,np.zeros(4),2,basis_coefficients=b)
    charge=dict.fromkeys(cells,0.)
    for section in re.split(r'(?=Section\[)',text):
        match=re.search(r'cell number: (\d+)',section)
        if not match:continue
        for expansion in re.findall(r'chargePA\[\d+\]:([^\n]*)',section):
            for weight,idx in re.findall(r'([-\d.eE+]+) B(\d+)\[',expansion):
                if int(idx)==index:charge[int(match[1])]+=float(weight)
    q=np.repeat([charge[c]/4 for c in cells],4)
    keep=(np.linalg.norm(mom,axis=1)>0)|(q!=0)
    return xyz[keep],mom[keep],q[keep]

k=2*np.pi*25e6/C
rows=[]
for i,j in [(9,24),(9,23),(10,23),(10,24),(9,12)]:
    x,m,q=basis(i);y,n,p=basis(j)
    dist=np.linalg.norm(x[:,None]-y[None,:],axis=2)
    green=np.exp(-1j*k*dist)/(4*np.pi*dist)
    vector=1j*k*ETA*np.sum(green*(m@n.T))
    scalar=-1j*ETA/k*np.sum(green*q[:,None]*p[None,:])
    rho=np.linalg.norm(x[:,None,:2]-y[None,:,:2],axis=2)
    lateral=np.exp(-1j*k*rho)/(4*np.pi*dist)
    lateral_z=1j*k*ETA*np.sum(lateral*(m@n.T))-1j*ETA/k*np.sum(lateral*q[:,None]*p[None,:])
    row=dict(basis=[i,j],vector_real=float(vector.real),scalar_real=float(scalar.real),
             total_real=float((vector+scalar).real),total_imag=float((vector+scalar).imag),
             lateral_phase_real=float(lateral_z.real),lateral_phase_imag=float(lateral_z.imag))
    rows.append(row);print(row)
(work/'interplane_kernel_diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')

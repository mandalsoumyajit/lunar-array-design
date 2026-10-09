"""Independent lossless half-space radiation from exported Momentum basis currents.

Restricted to horizontal, axis-aligned rectangular strip meshes. Uses the
solved spatial current distribution, not a field or power normalization fit.
"""
from pathlib import Path
import re
import numpy as np
from recover_response import C, ETA


def current_quadrature(work, port_current=1, order=4, basis_coefficients=None):
    work=Path(work)
    mesh=(work/'proj.msh').read_text()
    points={int(i):np.array(list(map(float,v.split()))) for i,v in re.findall(r'p (\d+): ([^,;]+)',mesh)}
    edges={int(i):(int(a),int(b)) for i,a,b in re.findall(r's (\d+): (\d+) (\d+)',mesh)}
    cells={int(i):list(map(int,v.split())) for i,v in re.findall(r'c (\d+): ([^,;]+)',mesh)}
    sections=(work/'BasisSections.log').read_text()
    feed={}
    for section in re.split(r'(?=Section\[)',sections):
        match=re.search(r'port number: (-?\d+)',section)
        if match:
            terms=re.findall(r'currentPA\[\d+\]: type: 0 ([^\n]+)',section)
            if len(terms)!=1:raise RuntimeError('Unsupported port basis expansion')
            term=re.fullmatch(r'\s*-1 B(\d+)\[0\]\s*',terms[0])
            if not term:raise RuntimeError('Unsupported port basis orientation')
            feed[int(term[1])-1]=int(match[1])
    currents=[]
    for name in sorted(work.glob('Imatrix.m.*'),key=lambda p:int(p.suffix[1:])):
        text=name.read_text()
        vals=re.findall(r'I\((\d+), 1\) = ([^ ]+) \+ j\* ([^;]+);',text)
        c=np.zeros(max(int(i) for i,_,_ in vals),complex)
        for i,a,b in vals:c[int(i)-1]=complex(float(a),float(b))
        currents.append(c)
    port_current=np.atleast_1d(port_current)
    terminal_basis=np.column_stack(currents)
    indices=sorted(feed)
    desired=np.array([-np.sign(feed[i])*port_current[abs(feed[i])-1] for i in indices])
    basis=terminal_basis@np.linalg.solve(terminal_basis[indices],desired)
    if basis_coefficients is not None:basis=np.asarray(basis_coefficients)
    coeff={i:np.zeros(4,complex) for i in cells}
    for section in re.split(r'(?=Section\[)',sections):
        match=re.search(r'cell number: (\d+)',section)
        if not match:continue
        cell=int(match[1])
        for edge,expansion in re.findall(r'currentPA\[(\d+)\]:([^\n]*)',section):
            for weight,idx in re.findall(r'([-\d.eE+]+) B(\d+)\[',expansion):
                coeff[cell][int(edge)]+=float(weight)*basis[int(idx)-1]
    nodes,weights=np.polynomial.legendre.leggauss(order); nodes=(nodes+1)/2;weights=weights/2
    xyz=[]; moments=[]
    for cell,edgeids in cells.items():
        vertices=np.array([points[n] for e in edgeids for n in edges[abs(e)]])
        lo,hi=vertices.min(axis=0),vertices.max(axis=0)
        if not (len(edgeids)==4 and hi[0]>lo[0] and hi[1]>lo[1] and hi[2]==lo[2]):
            raise RuntimeError('Current export supports only horizontal rectangular cells')
        if len(np.unique(vertices,axis=0))!=4 or not np.all((vertices==lo)|(vertices==hi)):
            raise RuntimeError('Rotated or nonrectangular cells are unsupported')
        area=(hi[0]-lo[0])*(hi[1]-lo[1]);center=(hi+lo)/2
        for u,wu in zip(nodes,weights):
            for v,wv in zip(nodes,weights):
                position=lo+(hi-lo)*[u,v,0]
                # Section current indices refer to global rectangle directions,
                # not the (orientation-dependent) order of edges in CdefIA.
                dx,dy=hi[:2]-lo[:2]
                j=np.array([(coeff[cell][0]*(1-u)-coeff[cell][2]*u)/dy,
                            (coeff[cell][1]*(1-v)-coeff[cell][3]*v)/dx,0],complex)
                xyz.append(position);moments.append(j*area*wu*wv)
    return np.array(xyz),np.array(moments)


def fields(xyz,moments,frequency,theta,phi,er=4,reference_z=0,layer_er=None,layer_depth=0):
    """Peak rE coefficient, e^(+jwt), observation angles in the actual medium."""
    theta,phi=np.broadcast_arrays(np.asarray(theta),np.asarray(phi))
    shape=theta.shape;theta=theta.ravel();phi=phi.ravel()
    scalar=moments.ndim==2
    if scalar:moments=moments[:,:,None]
    out=np.zeros((len(theta),2,moments.shape[2]),complex)
    k=2*np.pi*frequency/C;n=np.sqrt(er)
    if np.iscomplexobj(n) and np.any(theta>np.pi/2):
        raise ValueError('No far-field power interpretation in a lossy infinite lower medium; use upper fields and network dissipation')
    for start in range(0,len(theta),300):
        t=theta[start:start+300];p=phi[start:start+300]
        upper=t<=np.pi/2
        a=np.where(upper,t,np.pi-t);s=np.sin(a);c=np.cos(a)
        kt=k*s*np.where(upper,1,n)
        def kz(e):
            q=np.sqrt((e*k*k-kt*kt).astype(complex))
            return np.where(q.imag>0,q.conj(),q)
        q1=kz(1);q2=kz(er)
        q1=np.where(abs(q1)<1e-20,1e-20+0j,q1)
        phase_xy=np.exp(1j*(kt*np.cos(p))[:,None]*xyz[:,0]+1j*(kt*np.sin(p))[:,None]*xyz[:,1])
        if layer_er is None or layer_depth==0:
            rs=(q1-q2)/(q1+q2);rp=(q2-er*q1)/(q2+er*q1)
            ts=2*q2/(q2+q1);tp=2*er*q1/(er*q1+q2)
        else:
            ql=kz(layer_er);prop=np.exp(-1j*ql*layer_depth)
            def stack(y0,y1,y2):
                r01=(y0-y1)/(y0+y1);r12=(y1-y2)/(y1+y2)
                reflection=(r01+r12*prop**2)/(1+r01*r12*prop**2)
                transmission=(2*y2/(y2+y1))*(2*y1/(y1+y0))*prop/(1+r01*r12*prop**2)
                return reflection,transmission*np.exp(1j*q2*layer_depth)
            rs,ts=stack(q1,ql,q2)
            rp,tp=stack(1/q1,layer_er/ql,er/q2)
        te_up=np.exp(1j*q1[:,None]*xyz[:,2])+rs[:,None]*np.exp(-1j*q1[:,None]*xyz[:,2])
        tm_up=c[:,None]*(np.exp(1j*q1[:,None]*xyz[:,2])+rp[:,None]*np.exp(-1j*q1[:,None]*xyz[:,2]))
        te_down=ts[:,None]*np.exp(-1j*q1[:,None]*xyz[:,2])
        tm_down=(-c*tp)[:,None]*np.exp(-1j*q1[:,None]*xyz[:,2])
        te=np.where(upper[:,None],te_up,te_down)
        tm=np.where(upper[:,None],tm_up,tm_down)
        # Match the Momentum entering-port/field convention. The explicit
        # reference-plane phase changes no power, and applies in the observing medium.
        phase_origin=np.exp(-1j*k*np.where(upper,1,n)*np.cos(t)*reference_z)
        mt=phase_xy*tm;ms=phase_xy*te
        ft=(mt@moments[:,0,:])*np.cos(p)[:,None]+(mt@moments[:,1,:])*np.sin(p)[:,None]
        fs=-(ms@moments[:,0,:])*np.sin(p)[:,None]+(ms@moments[:,1,:])*np.cos(p)[:,None]
        out[start:start+len(t),0]=ft*(1j*k*ETA/(4*np.pi))*phase_origin[:,None]
        out[start:start+len(t),1]=fs*(1j*k*ETA/(4*np.pi))*phase_origin[:,None]
    out=out.reshape(shape+(2,moments.shape[2]))
    return out[...,0] if scalar else out


def integrate_power(xyz,moments,frequency,er=4,ntheta=80,nphi=120):
    # Split lower hemisphere at the critical angle to resolve the square-root cusp.
    bounds=[0,np.pi/2,np.pi-np.arcsin(1/np.sqrt(er)),np.pi]
    rows=[]
    for low,high in zip(bounds[:-1],bounds[1:]):
        x,w=np.polynomial.legendre.leggauss(ntheta)
        t=(x+1)*(high-low)/2+low;w=w*(high-low)/2*np.sin(t)
        p=np.arange(nphi)*2*np.pi/nphi
        ff=fields(xyz,moments,frequency,t[:,None],p[None,:],er)
        medium=1 if high<=np.pi/2 else np.sqrt(er)
        rows.append(float(np.sum(np.sum(abs(ff)**2,axis=-1)*w[:,None])*2*np.pi/nphi*medium/(2*ETA)))
    return rows


if __name__=='__main__':
    import keysight.ads.dataset as dataset
    work=Path(__file__).parent/'results/power_fix/debug_halfspace'
    with dataset.open_dataset_for_reading(work/'DebugField.ds') as ds:frame=ds['data'].to_dataframe()
    f=25e6;s=frame.at[f,'S[1,1]'];z=50*(1+s)/(1-s);cur=1/(z+50)
    xyz,mom=current_quadrature(work,cur,2)
    ff=fields(xyz,mom,f,np.deg2rad([0,45,90,135,180]),np.pi/2)
    print('Z/current',z,cur);print('fields',ff[:,1])
    powers=integrate_power(xyz,mom,f,ntheta=40,nphi=60)
    print('power components',powers,'accepted',0.5*abs(cur)**2*z.real,'ratio',sum(powers)/(0.5*abs(cur)**2*z.real))

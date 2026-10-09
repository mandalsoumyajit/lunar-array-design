"""Physical Galerkin station operator with balanced finite-gap source functions.

Uses the checked retarded-vacuum Momentum matrix for strip self interactions,
adds explicit impressed feed-current interactions, and evaluates the dielectric
reflection operator by a Sommerfeld plane-wave integral. No power fitting.
"""
from pathlib import Path
import re,json,argparse,shutil
import numpy as np
from scipy import sparse
from scipy.special import roots_legendre
from physical_halfspace_fields import current_quadrature,fields
from recover_response import C,ETA,digest

ROOT=Path(__file__).parent/'results/power_fix/station'

def prepare_vacuum(frequency,mesh_scale=1):
    """Generate a frequency-specific physical vacuum strip matrix, retaining it."""
    from momentum_safe import solve,verify_solve
    source=ROOT/'kernel_freespace'
    work=source if frequency==25e6 and mesh_scale==1 else ROOT/f'physical_reference_f{frequency/1e6:g}_mesh{mesh_scale:g}'
    if work!=source and not (work/'solve_manifest.json').exists():
        work.mkdir(exist_ok=True)
        for name in ['proj_a','proj.pin','proj.prt','proj.ltd','proj.opt','proj.params','proj.plan','proj.toi']:
            shutil.copy2(source/name,work/name)
        (work/'proj.sti').write_text(f'PT {frequency/1e9:.17g},\n;\n')
        path=work/'proj.opt';text=path.read_text()
        text=re.sub(r'no_cells_per_wavelength \d+;',f'no_cells_per_wavelength {round(240*25e6/frequency*mesh_scale)};',text)
        text=re.sub(r'DS_DIR "[^"]+";',lambda m:f'DS_DIR "{work.absolute()}";',text)
        path.write_text(text)
        import momentum_safe
        original_configure=momentum_safe.configure
        def configure_reference(*args,**kwargs):
            original_configure(*args,**kwargs)
            path=work/'momentum.cfg'
            path.write_text(path.read_text()+'MOM3D_USE_GFFREESPACE=TRUE\n')
        momentum_safe.configure=configure_reference
        try:solve(work,[frequency],export_currents=True)
        finally:momentum_safe.configure=original_configure
    verify_solve(work,[frequency])
    if 'MOM3D_USE_GFFREESPACE=TRUE' not in (work/'momentum.cfg').read_text():
        raise RuntimeError('Physical vacuum reference setting missing')
    return work

def mesh_basis(work):
    text=(work/'proj.msh').read_text()
    points={int(i):np.array(list(map(float,v.split()))) for i,v in re.findall(r'p (\d+): ([^,;]+)',text)}
    edges={int(i):(int(a),int(b)) for i,a,b in re.findall(r's (\d+): (\d+) (\d+)',text)}
    cells={int(i):list(map(int,v.split())) for i,v in re.findall(r'c (\d+): ([^,;]+)',text)}
    n=int(re.search(r'N: (\d+)',(work/'Basis.m').read_text())[1]);ids={c:i for i,c in enumerate(cells)}
    coefficients=[sparse.lil_matrix((len(cells),n)) for _ in range(4)];feed={}
    for section in re.split(r'(?=Section\[)',(work/'BasisSections.log').read_text()):
        match=re.search(r'cell number: (\d+)',section)
        if not match:continue
        cell=ids[int(match[1])]
        for edge,expansion in re.findall(r'currentPA\[(\d+)\]:([^\n]*)',section):
            for weight,b in re.findall(r'([-\d.eE+]+) B(\d+)\[',expansion):
                coefficients[int(edge)][cell,int(b)-1]+=float(weight)
                port=re.search(r'port number: (-?\d+)',section)
                if port:feed[int(b)-1]=int(port[1])
    lo=[];hi=[]
    for c,edgeids in cells.items():
        v=np.array([points[p] for e in edgeids for p in edges[abs(e)]])
        low,high=v.min(0),v.max(0)
        assert len(edgeids)==4 and high[2]==low[2] and np.all((v==low)|(v==high))
        lo.append(low);hi.append(high)
    lo=np.array(lo);hi=np.array(hi)
    terminals=sorted(feed);internal=[i for i in range(n) if i not in feed]
    transform=sparse.lil_matrix((n,len(internal)+4))
    for i,p in feed.items():transform[i,abs(p)-1]=-np.sign(p)
    for j,i in enumerate(internal):transform[i,j+4]=1
    transform=transform.tocsr();coefficients=[(c.tocsr()@transform).tocsr() for c in coefficients]
    heights=np.empty(transform.shape[1])
    support=sum(abs(c) for c in coefficients).tocsc()
    for col in range(len(heights)):
        z=lo[support[:,col].indices,2];assert np.ptp(z)==0;heights[col]=z[0]
    return lo,hi,coefficients,transform,np.array(terminals),np.array(internal),heights

def quadrature(lo,hi,coefficients,order=4):
    x,w=roots_legendre(order);x=(x+1)/2;w=w/2
    u,v=np.meshgrid(x,x,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    u=u.ravel();v=v.ravel();ww=(wu*wv).ravel();nq=len(u);dx,dy=(hi-lo)[:,:2].T
    xyz=(lo[:,None,:]+(hi-lo)[:,None,:]*np.stack([u,v,np.zeros(nq)],axis=-1)).reshape(-1,3)
    repeat=sparse.kron(sparse.eye(len(lo)),np.ones((nq,1)),format='csr')
    c0,c1,c2,c3=[repeat@c for c in coefficients]
    jx=sparse.diags(np.repeat(dx,nq)*np.tile(ww*(1-u),len(lo)))@c0-sparse.diags(np.repeat(dx,nq)*np.tile(ww*u,len(lo)))@c2
    jy=sparse.diags(np.repeat(dy,nq)*np.tile(ww*(1-v),len(lo)))@c1-sparse.diags(np.repeat(dy,nq)*np.tile(ww*v,len(lo)))@c3
    return xyz,jx.tocsr(),jy.tocsr()

def gap_quadrature(heights,order=12):
    x,w=roots_legendre(order);y,wy=roots_legendre(4)
    xyz=[];mom=[]
    for port,(length,width) in enumerate([(.1,.02),(.1,.02),(.05,.01),(.05,.01)]):
        axis=port%2
        for u,wu in zip(x,w):
            for v,wv in zip(y,wy):
                pos=np.zeros(3);pos[axis]=u*length/2;pos[1-axis]=v*width/2;pos[2]=heights[port]
                m=np.zeros((3,4));m[axis,port]=-length*wu*wv/4
                xyz.append(pos);mom.append(m)
    return np.array(xyz),np.array(mom)

def gap_self(length,width,k,order=48):
    x,w=roots_legendre(order);u=(x+1)/2;w=w/2
    u=u[:,None];v=((x+1)/2)[None,:];weights=w[:,None]*w[None,:]
    value=0j
    for dx,dy in [(length*u,width*u*v),(length*u*v,width*u)]:
        distance=np.hypot(dx,dy)
        value+=np.sum(weights*(length-dx)*(width-dy)*np.exp(-1j*k*distance)/distance*(length*width*u))
    return value/(np.pi*width**2)

def vacuum_operator(work,geometry,frequency):
    lo,hi,coefficients,T,terminals,internal,heights=geometry
    raw=(work/'ZMatrix.m').read_text();n=T.shape[0];z=np.zeros((n,n),complex)
    for i,j,a,b in re.findall(r'Z\((\d+), (\d+)\) = ([^ ]+) \+ j\* ([^ ;]+)',raw):z[int(i)-1,int(j)-1]=complex(float(a),float(b))
    z+=np.triu(z,1).T;z[terminals,terminals]-=50
    a=np.asarray(T.T@(T.T@z.T).T)
    xyz,jx,jy=quadrature(lo,hi,coefficients)
    gapxyz,gapmom=gap_quadrature(heights);k=2*np.pi*frequency/C
    distance=np.linalg.norm(xyz[:,None,:]-gapxyz[None,:,:],axis=2)
    green=np.exp(-1j*k*distance)/(4*np.pi*distance)
    cross=jx.T@(green@gapmom[:,0,:])+jy.T@(green@gapmom[:,1,:])
    a[:,:4]+=1j*k*ETA*cross;a[:4,:]+=1j*k*ETA*cross.T
    distance=np.linalg.norm(gapxyz[:,None,:]-gapxyz[None,:,:],axis=2)
    np.fill_diagonal(distance,1)
    green=np.exp(-1j*k*distance)/(4*np.pi*distance)
    gg=gapmom[:,0,:].T@green@gapmom[:,0,:]+gapmom[:,1,:].T@green@gapmom[:,1,:]
    for p,(length,width) in enumerate([(.1,.02),(.1,.02),(.05,.01),(.05,.01)]):gg[p,p]=gap_self(length,width,k)
    a[:4,:4]+=1j*k*ETA*gg
    # Recompute the smooth imaginary Green kernel directly. Small absolute
    # errors in exported scalar/vector potentials otherwise dominate dark port
    # combinations at 1 MHz. This is a real-space EFIE integral, independent of
    # any far-field, accepted-power result, or fitted normalization.
    a=vacuum_resistance_efie(geometry,frequency)+1j*a.imag
    return a


def vacuum_resistance_efie(geometry,frequency,order=3):
    lo,hi,cs,T,terminals,internal,heights=geometry
    xyz,jx,jy=quadrature(lo,hi,cs,order)
    x,w=roots_legendre(order);ww=np.outer(w/2,w/2).ravel()
    q=sparse.kron(sparse.eye(len(lo)),ww[:,None],format='csr')@(-sum(cs))
    if np.max(abs(np.asarray(q.sum(axis=0))))>1e-10:
        raise RuntimeError('Composite feed/basis has unbalanced charge')
    gapxyz,gapmom=gap_quadrature(heights)
    def augment(j,axis):
        gap=sparse.hstack([sparse.csr_matrix(gapmom[:,axis,:]),sparse.csr_matrix((len(gapxyz),j.shape[1]-4))])
        return sparse.vstack([j,gap],format='csr')
    jx=augment(jx,0);jy=augment(jy,1)
    q=sparse.vstack([q,sparse.csr_matrix((len(gapxyz),q.shape[1]))],format='csr')
    xyz=np.concatenate([xyz,gapxyz]);k=2*np.pi*frequency/C
    result=np.zeros((T.shape[1],T.shape[1]))
    # Block the smooth sinc kernel to bound memory; value at R=0 is analytic.
    for start in range(0,len(xyz),384):
        sl=slice(start,start+384)
        distance=np.linalg.norm(xyz[sl,None,:]-xyz[None,:,:],axis=2)
        kr=k*distance;green=np.sinc(kr/np.pi)
        result+=np.asarray(jx[sl].T@(green@jx)+jy[sl].T@(green@jy))*ETA*k*k/(4*np.pi)
        # Neutral basis functions annihilate the constant scalar kernel.
        # Use its series near zero to avoid subtracting nearly equal numbers.
        remainder=np.where(abs(kr)<.01,-kr**2/6+kr**4/120-kr**6/5040,green-1)
        result-=np.asarray(q[sl].T@(remainder@q))*ETA/(4*np.pi)
    return (result+result.T)/2

def unit_linear(t):
    a0=np.exp(.5j*t)*np.sinc(t/(2*np.pi))
    safe=np.where(abs(t)<1e-3,1,t)
    a1=a0/2-1j*np.exp(.5j*t)*(t*np.cos(t/2)-2*np.sin(t/2))/safe**2
    small=abs(t)<1e-3
    a1=np.where(small,.5+1j*t/3-t*t/8-1j*t**3/30+t**4/144,a1)
    return a0,a1

def fourier_currents(geometry,kx,ky):
    lo,hi,cs,T,terminals,internal,heights=geometry;dx,dy=(hi-lo)[:,:2].T
    ax,bx=unit_linear(kx[:,None]*dx);ay,by=unit_linear(ky[:,None]*dy)
    phase=np.exp(1j*(kx[:,None]*lo[:,0]+ky[:,None]*lo[:,1]))
    fx=(cs[0].T@(phase*dx*ay*(ax-bx)).T-cs[2].T@(phase*dx*ay*bx).T).T
    fy=(cs[1].T@(phase*dy*ax*(ay-by)).T-cs[3].T@(phase*dy*ax*by).T).T
    for port,(length,width) in enumerate([(.1,.02),(.1,.02),(.05,.01),(.05,.01)]):
        along,across=(kx,ky) if port%2==0 else (ky,kx)
        (fx if port%2==0 else fy)[:,port]-=length*np.sinc(along*length/(2*np.pi))*np.sinc(across*width/(2*np.pi))
    return fx,fy

def spectral_nodes(frequency,er,layer_er,depth,order,nphi):
    """Sommerfeld quadrature, including propagating and evanescent sectors."""
    k=2*np.pi*frequency/C
    bounds=sorted(set([0,k,k*np.sqrt(layer_er.real),k*np.sqrt(er.real),max(24.,4*k)]))
    phi=np.arange(nphi)*2*np.pi/nphi
    def kz(e,p):
        q=np.sqrt(complex(e*k*k-p*p));return q.conjugate() if q.imag>0 else q
    for low,high in zip(bounds[:-1],bounds[1:]):
        x,w=roots_legendre(order);t=(x+1)*np.pi/4
        # Remove square-root endpoint singularities on both sides.
        p=low+(high-low)*np.sin(t)**2;w=w*np.pi/4*(high-low)*np.sin(2*t)
        for pp,ww in zip(p,w):
            q=kz(1,pp);ql=kz(layer_er,pp);qb=kz(er,pp);prop=np.exp(-2j*ql*depth)
            def reflect(y0,y1,y2):
                r01=(y0-y1)/(y0+y1);r12=(y1-y2)/(y1+y2)
                return (r01+r12*prop)/(1+r01*r12*prop)
            yield pp,pp*ww*(2*np.pi/nphi)/(2*np.pi)**2,phi,q,reflect(q,ql,qb),reflect(1/q,layer_er/ql,er/qb)


def reflection_operator(geometry,frequency,er=4.4,layer_er=3.4,depth=1,order=32,nphi=96):
    k=2*np.pi*frequency/C;heights=geometry[-1]-1
    a=np.zeros((len(heights),len(heights)),complex)
    for pp,weight,phi,q,rs,rp in spectral_nodes(frequency,er,layer_er,depth,order,nphi):
        cos=np.cos(phi);sin=np.sin(phi)
        fx,fy=fourier_currents(geometry,pp*cos,pp*sin)
        te=-sin[:,None]*fx+cos[:,None]*fy;tm=cos[:,None]*fx+sin[:,None]*fy
        h=np.exp(-1j*q*heights)
        factor=k*ETA/(2*q)*weight
        a+=factor*((te*h).T@(te.conj()*h)*rs+(tm*h).T@(tm.conj()*h)*(rp*q*q/(k*k)))
    return (a+a.T)/2


def surface_flux(geometry,solution,frequency,er,layer_er,order=48,nphi=128):
    """Direct downward Poynting flux at z=0; NOT accepted minus sky power.

    In lossy ground this is total material absorption in the infinite stack.
    In lossless ground it is transmitted radiation, checked against lower fields.
    Returned matrix is twice peak-current power, in ohms.
    """
    k=2*np.pi*frequency/C;heights=geometry[-1]-1;total=np.zeros((4,4),complex)
    for pp,weight,phi,q,rs,rp in spectral_nodes(frequency,er,layer_er,1,order,nphi):
        fx,fy=fourier_currents(geometry,pp*np.cos(phi),pp*np.sin(phi));h=np.exp(-1j*q*heights)
        te=((-np.sin(phi)[:,None]*fx+np.cos(phi)[:,None]*fy)*h)@solution
        tm=((np.cos(phi)[:,None]*fx+np.sin(phi)[:,None]*fy)*h)@solution
        ws=abs(k*ETA/(2*q))**2*np.real((1+rs)*np.conj(q/(k*ETA)*(1-rs)))
        wp=abs(ETA*q/(2*k))**2*np.real((1+rp)*np.conj(k/(ETA*q)*(1-rp)))
        total+=weight*(ws*(te.conj().T@te)+wp*(tm.conj().T@tm))
    return (total+total.conj().T)/2


def run():
    p=argparse.ArgumentParser();p.add_argument('--layered',action='store_true');p.add_argument('--lossy',action='store_true')
    p.add_argument('--frequency-mhz',type=float,default=25);p.add_argument('--mesh-scale',type=float,default=1)
    p.add_argument('--order',type=int,default=32);p.add_argument('--nphi',type=int,default=96);p.add_argument('--field-order',type=int,default=48)
    p.add_argument('--tag',default='');args=p.parse_args()
    if args.lossy:args.layered=True
    if args.tag and not re.fullmatch('[a-z0-9_]+',args.tag):raise ValueError('Invalid output tag')
    if not 1<=args.frequency_mhz<=50 or not 1<=args.mesh_scale<=2:raise ValueError('Supported frequency scope is 1--50 MHz')
    f=args.frequency_mhz*1e6;work=prepare_vacuum(f,args.mesh_scale)
    label='physical_'+('lossy' if args.lossy else 'layered' if args.layered else 'vacuum')+f'_f{args.frequency_mhz:g}'
    if args.tag:label+='_'+args.tag
    out=ROOT/label;out.mkdir(exist_ok=True)
    geometry=mesh_basis(work)
    if not np.allclose(geometry[-1][:4],[7,7,1.5,1.5]):raise RuntimeError('Unexpected station/feed heights')
    provenance={str(path.absolute()):digest(path) for path in [Path(__file__),Path(__file__).with_name('physical_halfspace_fields.py'),work/'solve_manifest.json',work/'ZMatrix.m',work/'Basis.m',work/'BasisSections.log',work/'proj.msh']}
    cache=work/'physical_vacuum_operator.npz';cachekey=json.dumps(provenance,sort_keys=True)
    if cache.exists():
        with np.load(cache) as d:
            a=d['matrix'] if str(d['key'])==cachekey else None
    else:a=None
    if a is None:
        a=vacuum_operator(work,geometry,f);np.savez_compressed(cache,matrix=a,key=cachekey)
    er=4.4*(1-.01j) if args.lossy else 4.4 if args.layered else 1
    layer=3.4*(1-.005j) if args.lossy else 3.4 if args.layered else 1
    if args.layered:a+=reflection_operator(geometry,f,er=er,layer_er=layer,order=args.order,nphi=args.nphi)
    print(label,'operator ready',flush=True)
    solution=np.vstack([np.eye(4),-np.linalg.solve(a[4:,4:],a[4:,:4])])
    z=a[:4]@solution
    normal_basis=geometry[3]@solution
    moments=[]
    for port in range(4):
        xyz,j=current_quadrature(work,np.zeros(4),2,basis_coefficients=normal_basis[:,port]);moments.append(j)
    gapxyz,gapmom=gap_quadrature(geometry[-1]);xyz=np.concatenate([xyz,gapxyz]);mom=np.concatenate([np.stack(moments,axis=-1),gapmom])
    xyz[:,2]-=1
    bounds=[0,np.pi/2] if args.lossy else sorted(set([0,np.pi/2,np.pi-np.arcsin(1/np.sqrt(er)),np.pi-np.arcsin(np.sqrt(layer/er)),np.pi]))
    total=np.zeros((4,4),complex);upper=None
    for low,high in zip(bounds[:-1],bounds[1:]):
        x,w=roots_legendre(args.field_order);theta=(x+1)*(high-low)/2+low;w=w*(high-low)/2*np.sin(theta);phi=np.arange(2*args.field_order)*np.pi/args.field_order
        g=fields(xyz,mom,f,theta[:,None],phi[None,:],er=er,layer_er=layer,layer_depth=1)
        contribution=np.einsum('abip,abiq,a->pq',g.conj(),g,w)*np.pi/args.field_order/ETA*(1 if high<=np.pi/2 else np.sqrt(er))
        total+=contribution
        if high==np.pi/2:upper=contribution.copy()
    below=surface_flux(geometry,solution,f,er,layer,max(48,args.order),max(128,args.nphi))
    lower_error=None if args.lossy else float(np.linalg.norm(below-(total-upper))/np.linalg.norm(below))
    if args.lossy:total+=below
    resistance=(z+z.conj().T)/2;val,u=np.linalg.eigh(resistance)
    if val.min()<=0:raise RuntimeError('Nonpassive corrected network')
    ri=(u/np.sqrt(val))@u.conj().T
    eigen=np.linalg.eigvalsh(ri@total@ri);error=float(np.max(abs(eigen-1)))
    result=dict(frequency_hz=f,layered=args.layered,lossy=args.lossy,order=args.order,nphi=args.nphi,field_order=args.field_order,mesh_scale=args.mesh_scale,
                power_eigenvalues=eigen.tolist(),worst_power_error=error,lower_farfield_vs_surface_flux_relative_error=lower_error,
                reciprocity_relative_error=float(np.linalg.norm(z-z.T)/np.linalg.norm(z)),minimum_resistance_eigenvalue_ohm=float(val.min()),
                minimum_surface_flux_eigenvalue_ohm=float(np.linalg.eigvalsh(below).min()),
                passed_power_closure=bool(error<.001 and val.min()>0 and (lower_error is None or lower_error<.001)),
                model='PEC strips; ideal balanced finite-gap impressed feed currents; planar 1 m layer; no cables/supports',
                validated_for_science=False,provenance=provenance,z_real=z.real.tolist(),z_imag=z.imag.tolist())
    np.savez_compressed(out/'solution.npz',frequency_hz=f,z=z,basis=normal_basis,composite=solution,xyz=xyz,moments=mom,radiation=total,upper=upper,below=below)
    result['solution_sha256']=digest(out/'solution.npz')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()

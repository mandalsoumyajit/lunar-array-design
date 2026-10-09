"""Verify the observed lateral-distance phase error and its matrix consequence."""
from pathlib import Path
import re,json
import numpy as np
from recover_response import C,digest

root=Path(__file__).parent/'results/power_fix/station'
work=root/'kernel_gf_diagnostics'
text=(work/'gfFields_F1.m').read_text()
assert '%Row_3 = Ge_stripT2_stripT3' in text
assert '%Row_4 = Gm_stripT2_stripT3' in text
samples={int(i):float(x) for i,x in re.findall(r'rho\(1,(\d+)\)=([^;]+);',text)}
values={(kind,int(i),int(j)):complex(float(a),float(b)) for kind,i,j,a,b in
    re.findall(r'MGF_(images|rem)\((\d+),(\d+)\)=([^\n]+)\+j\*\(([^)]+)\);',text)}
indices=sorted(samples);rho=np.array([samples[i] for i in indices]);distance=np.hypot(rho,5.5)
k=2*np.pi*25e6/C
checks=[];arrays={}
for row in [3,4]:
    native=np.array([values['images',row,i]+values['rem',row,i] for i in indices])
    # The exported static image coefficient fixes C; this is not a fit.
    coefficient=values['images',row,1]*5.5
    physical=coefficient*np.exp(-1j*k*distance)/distance
    lateral=coefficient*np.exp(-1j*k*rho)/distance
    error=np.linalg.norm(native-lateral)/np.linalg.norm(native)
    assert error<1e-5
    checks.append(dict(row=row,lateral_phase_relative_l2=float(error),
        physical_phase_relative_l2=float(np.linalg.norm(native-physical)/np.linalg.norm(physical)),samples=len(rho)))
    arrays.update({f'native_row{row}':native,f'physical_row{row}':physical,f'lateral_row{row}':lateral})
np.savez_compressed(work/'kernel_phase_comparison.npz',rho_m=rho,distance_m=distance,**arrays)

def matrix(path):
    return {(int(i),int(j)):complex(float(a),float(b)) for i,j,a,b in re.findall(
        r'Z\((\d+), (\d+)\) = ([^ ]+) \+ j\* ([^ ;]+)',path.read_text())}
bad=matrix(root/'lossless_f25_vacuum_noportsolver/ZMatrix.m')
corrected=matrix(root/'kernel_freespace/ZMatrix.m')
rows=json.loads((root/'lossless_f25_vacuum_noportsolver/interplane_kernel_diagnostic.json').read_text())
interaction_checks=[]
for row in rows[:4]:
    pair=tuple(row['basis']);physical=complex(row['total_real'],row['total_imag'])
    lateral=complex(row['lateral_phase_real'],row['lateral_phase_imag'])
    bad_error=abs(bad[pair]-lateral)/abs(bad[pair]);corrected_error=abs(corrected[pair]-physical)/abs(physical)
    assert bad_error<2e-4 and corrected_error<2e-4
    interaction_checks.append(dict(basis=pair,native=[bad[pair].real,bad[pair].imag],
        independent_physical=[physical.real,physical.imag],independent_lateral=[lateral.real,lateral.imag],
        native_matches_lateral_relative=bad_error,diagnostic_freespace_matches_physical_relative=corrected_error))
result=dict(frequency_mhz=25,height_separation_m=5.5,missing_phase_at_zero_lateral_degrees=float(np.rad2deg(k*5.5)),
    kernel_checks=checks,interaction_checks=interaction_checks,
    finding='Observed vacuum interplane Green kernel uses exp(-j*k*rho)/R instead of exp(-j*k*R)/R.',
    limitation='Diagnostic free-space switch has not been validated for dielectric station; smaller feed-gap-dependent residual remains.',
    evidence_sha256={str(p.relative_to(root)):digest(p) for p in [work/'gfFields_F1.m',root/'lossless_f25_vacuum_noportsolver/ZMatrix.m',root/'kernel_freespace/ZMatrix.m']})
# Check the same directly exported interplane magnetic potential in the actual
# lossless dielectric stack. Other layered terms are not free-space kernels.
layered_path=root/'kernel_gf_diagnostics_layered/gfFields_F1.m'
text=layered_path.read_text();assert '%Row_4 = Gm_stripT2_stripT3' in text
lrho={int(i):float(x) for i,x in re.findall(r'rho\(1,(\d+)\)=([^;]+);',text)}
lval={(kind,int(i),int(j)):complex(float(a),float(b)) for kind,i,j,a,b in
    re.findall(r'MGF_(images|rem)\((\d+),(\d+)\)=([^\n]+)\+j\*\(([^)]+)\);',text)}
ids=[i for i in sorted(lrho) if ('images',4,i) in lval and lrho[i]>.01]
image=np.array([lval['images',4,i] for i in ids]);native=image+np.array([lval['rem',4,i] for i in ids])
lateral=image*np.exp(-1j*k*np.array([lrho[i] for i in ids]))
error=float(np.linalg.norm(native-lateral)/np.linalg.norm(native));assert error<1e-5
result['layered_interplane_magnetic_potential_lateral_phase_relative_l2']=error
result['layered_samples']=len(ids)
result['evidence_sha256'][str(layered_path.relative_to(root))]=digest(layered_path)
result['power_controls']={}
for name in ['freespace','smallgap_freespace','gf_diagnostics_layered','freespace_layered']:
    d=json.loads((root/('kernel_'+name)/'experiment_validation.json').read_text())
    result['power_controls'][name]=d['generalized_eigenvalues']
(work/'phase_kernel_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

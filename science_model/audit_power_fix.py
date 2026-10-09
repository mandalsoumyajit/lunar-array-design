"""Audit the correction evidence and rejection paths without a licensed solve."""
import ast,json
from pathlib import Path
from momentum_safe import database_manifest,verify_solve
from recover_response import digest

root=Path(__file__).parent;results=root/'results/power_fix'
controls=[]
for name in ['controls','controls_fine']:
    record=json.loads((results/name/'validation.json').read_text())
    assert record['passed'],name
    controls+=record['checks']
    for manifest in (results/name).rglob('solve_manifest.json'):
        saved=json.loads(manifest.read_text());verify_solve(manifest.parent,saved['requested_hz'])
try:
    database_manifest(results/'rejected_incomplete_database',[1e6,5e6,25e6])
except RuntimeError as exc:
    assert 'does not cover requested band' in str(exc),str(exc)
else:raise AssertionError('Incomplete database was accepted')
stations={}
for name in ['lossless_f25','lossless_f25_vacuum_noportsolver','nominal_f25']:
    d=json.loads((results/'station'/name/'validation.json').read_text())
    assert not d['passed'] and not d['validated_for_science'],name
    assert digest(results/'station'/name/'response.npz')==d['response_sha256']
    for file,expected in d['code_sha256'].items():assert digest(root/file)==expected,file
    stations[name]={key:d[key] for key in ['passed','validated_for_science','maximum_absolute_upper_field_relative_error']}
original=json.loads((root/'results/response_recovery/provenance.json').read_text())
proposal=root.parents[2]/'external/em_archive'
count=0
for group in original['inputs']:
    for name,expected in group.items():
        assert digest(proposal/name)==expected,name
        count+=1
scripts=['momentum_safe.py','physical_halfspace_fields.py','correct_station_response.py','validate_power_fix.py',
         'diagnose_terminal_power.py','check_scale_invariance.py','plot_revision_evidence.py','audit_power_fix.py']
for name in scripts:ast.parse((root/name).read_text())
audit=dict(isolated_controls_pass=True,isolated_control_count=len(controls),
    maximum_control_power_error=max(abs(r['power_ratio']-1) for r in controls),
    maximum_control_upper_field_error=max(r['absolute_upper_field_relative_error'] for r in controls),
    incomplete_database_rejected=True,original_files_unchanged=count,station_checks=stations,
    source_sha256={name:digest(root/name) for name in scripts},
    scope='Correction audit passes; station physical validation does not pass.')
(results/'fix_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))

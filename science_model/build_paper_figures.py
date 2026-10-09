"""Rebuild all data figures and consolidate their final provenance.
Runs component drivers in dependency order; EM cases must already exist.
"""
from pathlib import Path
import subprocess,sys,json
from recover_response import digest
ROOT=Path(__file__).parent
scripts=['fso_exact_capture.py','plot_paper_results.py','summarize_high_band.py','paper_design_decisions.py','causal_timing_check.py']
for name in scripts:
 result=subprocess.run([sys.executable,str(ROOT/name)],capture_output=True,text=True)
 if result.returncode:raise RuntimeError(name+'\n'+result.stdout+result.stderr)
 print('Completed '+name,flush=True)
base=ROOT/'results/paper_evidence/figure_validation.json'
original=json.loads(base.read_text());preserved=base.with_name('base_figure_validation.json');preserved.write_text(json.dumps(original,indent=2)+'\n')
records=[preserved,ROOT/'results/high_band_validation/validation.json',ROOT/'results/design_decisions/validation.json',ROOT/'results/design_decisions/causal_timing_validation.json']
# Verify input hashes in every study. Base figure hashes describe intermediate
# images; the final image hashes below cover replacements from the design study.
for p in records:
 obj=json.loads(p.read_text())
 for rel,h in obj.get('inputs_sha256',{}).items():
  assert digest(ROOT.parent/rel)==h,rel
figures=['science_brightness','corrected_station_response','timing_results','receiver_requirements','high_band_validation','array_layout_results','calibration_results']
manifest=dict(status='Final seven data figures; two additional figures are native TikZ diagrams',drivers=[str(ROOT/name) for name in scripts],inputs_sha256={str(p.relative_to(ROOT.parent)):digest(p) for p in [Path(__file__),*[ROOT/n for n in scripts],*records,ROOT.parent/'figures/station_architecture.tex',ROOT.parent/'figures/deployment_workflow.tex']},figures_sha256={name:digest(ROOT.parent/'figures'/(name+'.png')) for name in figures},note='Base bundle is intermediate. Final receiver/layout outputs and the high-band output are validated by their own retained study records.')
base.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print('Consolidated final figure manifest',flush=True)

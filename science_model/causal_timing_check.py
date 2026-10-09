"""A causal thermal correction tested on a changed synthetic temperature history."""
from pathlib import Path
import importlib.util,sys,json,numpy as np
from recover_response import digest,write_csv
ROOT=Path(__file__).parent;OUT=ROOT/'results/design_decisions'
p=ROOT/'results/paper_evidence/simulate_fiber_thermal_asymmetry.py'
spec=importlib.util.spec_from_file_location('fiber_reference',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
t=np.arange(0,24,1/60);rng=np.random.default_rng(20261009)
def past_average(x,n=15):
 return np.convolve(np.pad(x,(n-1,0),mode='edge'),np.ones(n)/n,mode='valid')
def features(sensors,length):
 a,b,c=[past_average(x)-250 for x in sensors];static=np.column_stack([np.ones(len(a)),length*a,length*b,c]);der=[np.r_[0,np.diff(x)]*60 for x in [a,b,c]]
 return np.column_stack([static,length*der[0],length*der[1],der[2],c*c])
def scenario(changed):
 temps=m.thermal_profiles(t) if not changed else tuple(250+scale*(x-250)+offset for x,scale,offset in zip(m.thermal_profiles(1.3*t+3.7),[.85,1.1,.9],[5,-3,4]))
 sm=m.SensorModel();sensors=[m.first_order_sensor(x,60,lag,noise,rng) for x,lag,noise in zip(temps,[sm.cable_lag_minutes,sm.sfp_lag_minutes,sm.board_lag_minutes],[sm.cable_noise_k,sm.sfp_noise_k,sm.board_noise_k])]
 noise=rng.normal(0,4,len(t));return temps,sensors,noise
train=scenario(False);test=scenario(True);rows=[]
for length in [.1,1,10]:
 def series(sc):
  (ca,sf,bo),sensors,noise=sc
  y=-.2*length*(ca-250)-.51*length*(sf-250)+10.85*(bo-250)+.012*(bo-250)**2+noise
  return y,features(sensors,length)
 y,X=series(train);v,V=series(test);valid=t>=1
 assert np.allclose(features([a[:720] for a in test[1]],length),V[:720],rtol=0,atol=1e-10)
 coef=np.linalg.lstsq(X[valid],y[valid],rcond=None)[0];res=v[valid]-V[valid]@coef
 row=dict(length_km=length,uncorrected_centered_rms_ps=float(np.std(v[valid])),causal_residual_rms_ps=float(np.sqrt(np.mean(res**2))),causal_residual_mean_ps=float(np.mean(res)),causal_centered_rms_ps=float(np.std(res)));rows.append(row)
write_csv(OUT/'causal_timing.csv',rows)
report=dict(check='Prefix invariance passed: features use no future samples',seed=20261009,training_hours=24,test_hours=24,startup_excluded_hours=1,changed_history='Test time coordinate 1.3*t+3.7 hours; cable/SFP/board deviations scaled by .85/1.1/.9 with offsets +5/-3/+4 K; unchanged physical coefficients and sensor laws.',noise_floor_ps=4,smoothing='15-sample trailing mean at 1-minute cadence; backward differences; no alignment shift',results=rows,inputs_sha256={str(q.relative_to(ROOT.parent)):digest(q) for q in [p,Path(__file__),OUT/'causal_timing.csv']})
(OUT/'causal_timing_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows,indent=2))

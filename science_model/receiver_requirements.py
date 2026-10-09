"""Full four-port receiver requirements from the repaired nominal station.

Uniform-sky comparison, not a site-specific sensitivity or hardware design.
All capacitances/noises refer to the complete differential antenna port.

Criterion (2026-10-09): receiver-added noise (amplifier, bias and series
protection) must stay below TARGET times the sky noise. Regolith emission is an
antenna-temperature term and is reported separately (ground_over_sky). Its
default brightness, 250 K, is the subsurface regolith temperature from which
14-50 MHz emission originates; the 100 K night surface value is not used.
"""
import csv,json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh
from response_bridge import load_physical_station_response
from physical_station_solver import ROOT as STATION
from recover_response import KB,digest

ROOT=Path(__file__).parent
OUT=ROOT/'results/receiver_requirements'
FREQUENCIES=[1,5,10,14,15,20,25,30,35,40,45,46,50]
TARGET=.25


def hermitian(a):return (a+a.conj().T)/2


def circuit(z,frequency,capacitance_pf,resistance_ohm):
    y=np.diag(1/np.broadcast_to(resistance_ohm,4)+2j*np.pi*frequency*np.broadcast_to(capacitance_pf,4)*1e-12)
    loading=np.linalg.solve(np.eye(4)+z@y,np.eye(4))
    parallel=loading@z
    return loading,parallel,y


def noise_model(z,rs,rg,frequency,capacitance_pf,en_nv=3,in_pa=0,resistance_ohm=1e6,
                sky_scale=1,ground_k=250,bias_k=300,rho=0,series_ohm=0):
    """e: additive amplifier input-referred noise; i: current emitted at terminal.

    Cei=E[e i^H]; current is positive into the antenna/input network.
    The sign and correlation must use this convention in measured device models.
    """
    if abs(rho)>1:raise ValueError('Nonphysical voltage/current noise correlation')
    series=np.diag(np.broadcast_to(series_ohm,4))
    L,P,Y=circuit(z+series,frequency,capacitance_pf,resistance_ohm)
    ts=sky_scale*50000*(frequency/20e6)**-2.5
    sky=4*KB*ts*L@rs@L.conj().T
    ground=4*KB*ground_k*L@rg@L.conj().T
    bias=4*KB*bias_k*P@Y.real@P.conj().T
    protection=4*KB*bias_k*L@series@L.conj().T
    en=np.broadcast_to(np.asarray(en_nv)*1e-9,4);ino=np.broadcast_to(np.asarray(in_pa)*1e-12,4)
    ce=np.diag(en**2);ci=np.diag(ino**2);cei=np.diag(rho*en*ino)
    current=P@ci@P.conj().T
    cross=cei@P.conj().T+P@cei.conj().T
    receiver=ce+current+cross
    return dict(loading=L,parallel=P,sky=hermitian(sky),ground=hermitian(ground),bias=hermitian(bias),
                voltage=ce,current=hermitian(current),cross=hermitian(cross),receiver=hermitian(receiver),
                protection=hermitian(protection),nonsky=hermitian(bias+protection+receiver),
                total_nonsky=hermitian(ground+bias+protection+receiver),sky_temperature_k=ts)


def select(a,ports):return a[np.ix_(ports,ports)]


def metrics(model,ports):
    sky=select(model['sky'],ports);other=select(model['nonsky'],ports)
    modes=eigvalsh(other,sky)
    total=sky+other
    ground_modes=eigvalsh(select(model['ground'],ports),sky);total_modes=eigvalsh(select(model['total_nonsky'],ports),sky)
    return dict(worst_nonsky_over_sky=float(modes[-1]),best_nonsky_over_sky=float(modes[0]),
                ground_over_sky=float(ground_modes[-1]),worst_total_over_sky=float(total_modes[-1]),
                noise_power_penalty_db=float(10*np.log10(1+modes[-1])),passes_4_to_1=bool(modes[-1]<=TARGET),
                sky_node_asd_nv=float(np.sqrt(sky[0,0].real)*1e9),
                ground_node_asd_nv=float(np.sqrt(model['ground'][ports[0],ports[0]].real)*1e9),
                receiver_node_asd_nv=float(np.sqrt(max(0,model['receiver'][ports[0],ports[0]].real))*1e9),
                bias_node_asd_nv=float(np.sqrt(model['bias'][ports[0],ports[0]].real)*1e9),
                total_node_asd_nv=float(np.sqrt(total[0,0].real)*1e9))


def voltage_limit(model,ports):
    """Joint 4:1 receiver budget: common independent e_n, with modeled i_n retained."""
    budget=select(TARGET*model['sky']-model['bias']-model['protection']-model['current'],ports)
    margin=float(np.linalg.eigvalsh(hermitian(budget))[0])
    return np.sqrt(max(0,margin))*1e9,margin>0


def current_limit(model,ports,en_nv):
    budget=select(TARGET*model['sky']-model['bias']-model['protection'],ports)-en_nv**2*1e-18*np.eye(len(ports))
    coefficient=select(model['parallel']@model['parallel'].conj().T,ports)
    margin=float(eigvalsh(hermitian(budget),hermitian(coefficient))[0])
    return np.sqrt(max(0,margin))*1e12,margin>0


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate():
    z=np.diag([50,100,200,300]).astype(complex)
    L,P,Y=circuit(z,20e6,0,[50,100,200,300])
    assert np.allclose(L,.5*np.eye(4)) and np.allclose(P,z/2)
    z=np.array([[100+30j,20+10j,0,0],[20+10j,80-50j,0,0],[0,0,120+90j,8j],[0,0,8j,200-70j]])
    L,P,Y=circuit(z,25e6,[3,5,10,35],[1e5,1e6,1e7,1e6])
    assert np.allclose(P,np.linalg.inv(np.linalg.inv(z)+Y))
    R=hermitian(z)
    assert np.allclose(L@R@L.conj().T+P@Y.real@P.conj().T,hermitian(P))
    for rho in [0,1,-1,1j,-1j,.5+.5j]:
        m=noise_model(z,R,np.zeros((4,4)),25e6,10,en_nv=2,in_pa=.3,rho=rho)
        assert np.linalg.eigvalsh(m['receiver']).min()>-1e-30
        ce=np.eye(4)*4e-18;ci=np.eye(4)*.09e-24;cei=np.eye(4)*rho*2e-9*.3e-12
        joint=np.block([[ce,cei],[cei.conj().T,ci]])
        transfer=np.column_stack([np.eye(4),m['parallel']])
        assert np.allclose(m['receiver'],transfer@joint@transfer.conj().T,rtol=1e-12,atol=1e-30)
    # Requirements must land on the boundary, not merely mirror the sweep.
    m=noise_model(z,R,np.zeros((4,4)),25e6,10,en_nv=0,in_pa=.1)
    limit,ok=voltage_limit(m,[0,1]);assert ok
    boundary=noise_model(z,R,np.zeros((4,4)),25e6,10,en_nv=limit,in_pa=.1)
    assert abs(metrics(boundary,[0,1])['worst_nonsky_over_sky']-TARGET)<1e-12
    m=noise_model(z,R,np.zeros((4,4)),25e6,10,en_nv=0,in_pa=0,series_ohm=49.8)
    expected=4*KB*300*hermitian(m['parallel'])
    thermal=4*KB*300*m['loading']@R@m['loading'].conj().T+m['bias']+m['protection']
    assert np.allclose(thermal,expected,rtol=1e-12,atol=1e-30)
    return ['resistive divider','nonidentical multiport loads','thermal equilibrium',
            'joint correlated e/i covariance and positivity','voltage-noise requirement boundary','series protection thermal equilibrium']


def main():
    OUT.mkdir(exist_ok=True)
    checks=validate();inputs={};data={};baseline_error=[]
    for f in FREQUENCIES:
        path=STATION/f'physical_lossy_f{f}/response.npz'
        load_physical_station_response(path)
        with np.load(path) as d:
            z=d['z_ohm'];rs=d['sky_resistance_ohm'];rg=d['ground_resistance_ohm']
            baseline=noise_model(z,rs,rg,f*1e6,35,ground_k=100)  # exports were written with a 100 K ground
            baseline_error.append(float(np.linalg.norm(baseline['ground']+baseline['bias']+baseline['receiver']-d['ground_psd_v2_hz']-d['receiver_psd_v2_hz'])/np.linalg.norm(baseline['total_nonsky'])))
        data[f]=(z,rs,rg);inputs[path.resolve().relative_to(ROOT.parent.resolve()).as_posix()]=digest(path)
    assert max(baseline_error)<1e-12
    checks.append('reproduce all corrected reference noise exports')
    sweep=[];limits=[];summary=[];correlations=[];protection=[];impedances=[];thermal=[]
    caps=[3,5,10,20,35]
    # Sweep all inputs together; retain coupling and noise emitted by every port.
    for f,(z,rs,rg) in data.items():
        for a in range(4):
            for b in range(4):impedances.append(dict(frequency_mhz=f,row_port=a+1,column_port=b+1,z_real_ohm=z[a,b].real,z_imag_ohm=z[a,b].imag))
        for clf in caps:
            for chf in caps:
                for scale in [.3,1,3]:
                    for resistance in [1e6,10e6]:
                        base=noise_model(z,rs,rg,f*1e6,[clf,clf,chf,chf],en_nv=0,resistance_ohm=resistance,sky_scale=scale)
                        for band,ports in [('LF',[0,1]),('HF',[2,3])]:
                            enmax,feasible=voltage_limit(base,ports)
                            imax,ifeasible=current_limit(base,ports,1)
                            limits.append(dict(frequency_mhz=f,band=band,c_lf_pf=clf,c_hf_pf=chf,sky_scale=scale,bias_resistance_ohm=resistance,
                                               en_limit_nv_at_zero_in=enmax,voltage_budget_positive=feasible,
                                               in_limit_pa_at_en_1nv=imax,current_budget_positive=ifeasible))
                        for en in [.5,1,2,3]:
                            for ino in [0,.1,.3,1]:
                                current=ino**2*1e-24*base['parallel']@base['parallel'].conj().T
                                m={**base,'current':hermitian(current),'voltage':en**2*1e-18*np.eye(4)}
                                m['receiver']=m['voltage']+m['current'];m['nonsky']=base['bias']+base['protection']+m['receiver'];m['total_nonsky']=base['ground']+m['nonsky']
                                for band,ports in [('LF',[0,1]),('HF',[2,3])]:
                                    sweep.append(dict(frequency_mhz=f,band=band,c_lf_pf=clf,c_hf_pf=chf,en_nv=en,in_pa=ino,sky_scale=scale,
                                                      bias_resistance_ohm=resistance,**metrics(m,ports)))
        for name,cp,en,ino in [('reference',35,3,0),('capacitance_only',10,3,0),('screen_5pf_1nv',5,1,.1),('screen_3pf_1nv',3,1,.1),('screen_5pf_08nv',5,.8,.1),('screen_3pf_08nv',3,.8,.1),('prototype_target',3,.7,.1)]:
            for scale in [.3,1]:
                model=noise_model(z,rs,rg,f*1e6,cp,en,ino,sky_scale=scale)
                for band,ports in [('LF',[0,1]),('HF',[2,3])]:
                    limit,feasible=voltage_limit(model,ports)
                    summary.append(dict(case=name,frequency_mhz=f,band=band,capacitance_pf=cp,en_nv=en,in_pa=ino,
                                        sky_scale=scale,en_limit_nv_at_given_in=limit,voltage_budget_positive=feasible,**metrics(model,ports)))
        for phase in np.arange(8)*np.pi/4:
            model=noise_model(z,rs,rg,f*1e6,5,1,.3,rho=.5*np.exp(1j*phase))
            for band,ports in [('LF',[0,1]),('HF',[2,3])]:
                correlations.append(dict(frequency_mhz=f,band=band,rho_abs=.5,rho_phase_deg=float(np.rad2deg(phase)),**metrics(model,ports)))
        for series in [0,10,49.8]:
            for scale in [.3,1]:
                model=noise_model(z,rs,rg,f*1e6,5,.8,.1,series_ohm=series,sky_scale=scale)
                for band,ports in [('LF',[0,1]),('HF',[2,3])]:
                    limit,feasible=voltage_limit(model,ports)
                    protection.append(dict(frequency_mhz=f,band=band,series_total_differential_ohm=series,sky_scale=scale,
                                           en_limit_nv=limit,voltage_budget_positive=feasible,**metrics(model,ports)))
        if f>=10:
            # Isolate known passive temperature effects. Amplifier noise and C
            # remain REQUIREMENT LIMITS here, not predicted device T laws.
            for bias_c in [-173.15,-55,0,26.85,85,125]:
                for ground in [200,250,260]:
                    worst=-1.;phase_worst=0.
                    for phase in np.arange(72)*2*np.pi/72:
                        model=noise_model(z,rs,rg,f*1e6,3,.7,.1,series_ohm=10,sky_scale=.3,
                                          bias_k=bias_c+273.15,ground_k=ground,rho=np.exp(1j*phase))
                        ratio=metrics(model,[2,3])['worst_nonsky_over_sky']
                        if ratio>worst:worst=ratio;phase_worst=float(np.rad2deg(phase))
                    floor=noise_model(z,rs,rg,f*1e6,3,0,0,series_ohm=0,sky_scale=.3,
                                      bias_k=0,ground_k=ground)  # ground-only term
                    thermal.append(dict(frequency_mhz=f,bias_and_series_temperature_c=bias_c,ground_temperature_k=ground,
                                        c_diff_pf=3,en_limit_nv=.7,in_limit_pa=.1,series_total_ohm=10,sky_scale=.3,
                                        largest_tested_nonsky_over_sky=worst,rho_abs=1,rho_phase_deg=phase_worst,
                                        ground_only_over_sky=metrics(floor,[2,3])['ground_over_sky'],
                                        passes_4_to_1=bool(worst<=TARGET)))
    write_csv('four_port_sweep.csv',sweep);write_csv('noise_limits.csv',limits)
    write_csv('comparison.csv',summary);write_csv('correlation_screen.csv',correlations)
    write_csv('protection_screen.csv',protection);write_csv('antenna_impedance_matrix.csv',impedances)
    write_csv('passive_temperature_and_correlation_screen.csv',thermal)
    payload=dict(passed=True,checks=checks,baseline_maximum_relative_error=max(baseline_error),
                 frequencies_mhz=FREQUENCIES,sweep_rows=len(sweep),inputs=inputs,source_sha256=digest(Path(__file__)),
                 assumptions=dict(sky='Uniform upper hemisphere: scale*50000*(frequency/20MHz)^(-2.5) K; scales 0.3,1,3 are scenarios, not measured bounds',
                                  ground_k=250,bias_k=300,target='sky power >= 4*(all receiver + bias + protection noise), for both selected receptor modes; regolith emission reported separately as ground_over_sky',
                                  capacitance='Complete differential input, including package/protection/feed/PCB; all four loads change jointly',
                                  noise='Independent channels; e/i uncorrelated in main sweep; explicit rho screen uses |rho|=0.5 at eight phases',
                                  temperature_screen='Passive resistors only: -173.15 to +125 C; ground term at 200/250/260 K; C=3 pF, en=0.7 nV/rtHz and in=0.1 pA/rtHz held at requirement limits, series R=10 ohm, |rho|=1 at 72 common phases; not device temperature prediction or qualification',
                                  scope='Nominal ideal station at sparse frequency samples; no hardware feasibility, source-direction ranking, continuous-band or absolute science-performance claim'),
                 selected_mainband=[r for r in summary if r['band']=='HF' and r['frequency_mhz']>=10 and r['sky_scale']==1])
    (OUT/'validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ['inputs','selected_mainband']},indent=2))
    for case in ['reference','capacitance_only','prototype_target']:
        selected=[r for r in summary if r['case']==case and r['band']=='HF' and r['sky_scale']==.3 and 14<=r['frequency_mhz']<=46]
        worst=max(selected,key=lambda r:r['worst_nonsky_over_sky'])
        print(case,'dim-sky priority-band worst',worst['frequency_mhz'],worst['worst_nonsky_over_sky'])

if __name__=='__main__':main()

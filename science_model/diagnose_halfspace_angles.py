"""Compare exported lower-hemisphere angles with TE Fresnel reciprocity.

The dipole lies along x. At phi=90 degrees, k_x=0 at every angle, so its
current-moment factor cancels against the normal upper-hemisphere field.
The thin-strip approximation neglects transverse variation across 1 mm.
No scale is fitted to make integrated power close.
"""
import numpy as np
from recover_response import ROOT, C, parse_fields, write_csv


def main():
    root=ROOT/'results/power_investigation/halfspace'
    rows=[]
    for f in [5,25]:
        for height in [0,6]:
            angles,field=parse_fields(root/f'h{height}/fields/f{f}_r50_v1.fff')
            def at(theta):
                return np.linalg.norm(field[(angles[:,0]==theta)&(angles[:,1]==90)][0])
            k=2*np.pi*f*1e6/C
            upper_normal=abs(1-np.exp(2j*k*height)/3)
            for theta in [180,160,150,140,135,120,100]:
                a=np.deg2rad(180-theta)
                cos_air=np.sqrt(complex(1-4*np.sin(a)**2))
                physical=abs(4*np.cos(a)/(2*np.cos(a)+cos_air))*np.exp(-k*cos_air.imag*height)
                # Signature of using the printed angle as an AIR incidence angle.
                air_angle_signature=4*np.cos(a)/(np.cos(a)+np.sqrt(4-np.sin(a)**2))
                observed=at(theta)/at(0)*upper_normal
                rows.append(dict(frequency_mhz=f,height_m=height,printed_theta_deg=theta,
                    observed_relative_to_current_moment=float(observed),
                    physical_dielectric_observation_angle=float(physical),
                    air_incidence_angle_signature=float(air_angle_signature),
                    relative_signature_error=float(observed/air_angle_signature-1)))
    write_csv(root/'angular_diagnosis.csv',rows)
    print('Maximum fractional signature discrepancy:',max(abs(r['relative_signature_error']) for r in rows))


if __name__=='__main__':main()

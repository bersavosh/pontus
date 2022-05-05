import os
import numpy as np
import scipy.stats as st
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from IPython.display import display

import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
colorset = ['#000000','#00270C','#00443C','#005083',
            '#034BCA','#483CFC','#9C2BFF','#EB24F4',
            '#FF2DC2','#FF4986','#FF7356','#FFA443',
            '#EBD155','#D3F187','#D7FFC8','#FFFFFF']

import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, vstack
from astroquery.utils.tap.core import TapPlus
from astroquery.simbad import Simbad

from zero_point import zpt
zpt.load_tables()

def bailerjones(RA, DEC,parallax,parallax_er):
    c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # RA and DEC in degrees of the source
    l = c_icrs.galactic.l.deg                                         # Converting to Galactic longitude 
    b = c_icrs.galactic.b.deg                                         # Converting to Galactic latitude

    rscript = f"""# Input data
# Specify either l,b or rlen. Set other to NA. rlen takes precedence.
w    <- {parallax} # parallax in mas (corrected for any zeropoint offset; +0.029mas in the catalogue)
wsd  <- {parallax_er} # parallax uncertainty in mas
glon <- {l} # Galactic longitude in degrees (0 to 360)
glat <-  {b} # Galactic latitude (-90 to +90)
rlen <-  NA # length scale in pc
# Plotting parameters in pc
# rlo,rhi are range for computing normalization of posterior
# rplotlo, rplothi are plotting range (computed automatically if set to NA)
rlo <- 0
rhi <- 1e5
rplotlo <- NA
rplothi <- NA

source("distest_single.R")
"""
    f = open("rscript.r", "w")
    f.write(rscript)
    f.close()
    os.system('Rscript rscript.r')
    bj_post = ascii.read('bj_post.txt', names = ['d','pdf'])
    bj_results = ascii.read('bj_post_res.txt',data_start=0,names=['point_est'])
    bj_post['d'] /= 1000.0
    bj_post['pdf'] *= 1000.0
    bj_results['point_est'] /= 1000.0
    return bj_post, bj_results

def bailerjones_new(src_id):
    gaia = TapPlus(url="http://dc.zah.uni-heidelberg.de/tap",verbose=False)

    adql_query_newbj = f"""SELECT 
                                source_id,
                                r_med_geo, r_lo_geo, r_hi_geo,
                                r_med_photogeo, r_lo_photogeo, r_hi_photogeo 
                           FROM gedr3dist.main
                              WHERE source_id = {src_id}
                        """

    job = gaia.launch_job(adql_query_newbj)
    result = job.get_results()
    return result


def atri_dist(parallax, parallax_er, ra, dec):
    # Defining some global constants which do not change with source
    r_lim = 40 #kpc
    r0 = 0.01 #kpc
    
    #Normalisation constants for density prior
    rho_0b = 1.0719
    rho_0d = 2.6387
    rho_0s = 13.0976
    
    #Parameters for density prior
    q = 0.6
    gamma = 1.8
    rt = 1.9
    rm = 6.5
    rd = 3.5
    rz = 0.41
    bs = 7.669
    Re = 2.8
    
    
    w = parallax             # mas PARALLAX
    s = parallax_er          # mas PARALLAX ERROR
    RA = ra                  # RA in degrees
    DEC = dec                # Dec in degrees
    c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # RA and DEC in degrees of the source
    l = c_icrs.galactic.l.deg                                         # Converting to Galactic longitude 
    b = c_icrs.galactic.b.deg                                         # Converting to Galactic latitude
    
    
    def bulge(r,z):
        k = (r**2 + ((z**2)/(q**2)))
        return rho_0b*((np.sqrt(k))**(-gamma))*np.exp(-k/(rt**2))
    def disk(r,z):
        return rho_0d*(np.exp(-(rm/rd)-(r/rd)-(np.abs(z)/rz)))
    def sphere(R):
        return rho_0s*(np.exp(-bs*((R/Re)**(1.0/4)))/((R/Re)**(7.0/8.0)))
    
    def new_prior(x,l,b,r_lim):
        if x > 0 and x <= r_lim:
            z = x*np.sin(np.radians(b))
            R0 = 8 #Distance of earth from the Galactic centre
            r = np.sqrt(R0**2 + ((x*np.cos(np.radians(b)))**2) - 2*x*R0*np.cos(np.radians(b))*np.cos(np.radians(l)))
            R = np.sqrt(R0**2 + (x**2) - 2*x*R0*np.cos(np.radians(b))*np.cos(np.radians(l)))
            rho_b = bulge(r,z)
            rho_d = disk(r,z)
            rho_s = sphere(R)
            return (rho_b+rho_d+rho_s)*((x*1e3)**2)
        else:
            return 0
        
    def posterior(x,w,s,r_lim):
        return new_prior(x,l,b,r_lim)*likelihood(x,w,s)
    
    """LIKELIHOOD
    """
    def likelihood(x,w,s):
        """Return the likelihood
        
        ===========
        Parameters:
        ===========
            **x**
            : number
                Real distance in kpc units.
            **w**
            : number
                Observed parallax in mas units.
            **s** 
            : number
                Parallax error in mas units.
        
        ========
        Returns:
        ========
            **number**
                Returns the likelihood.
        """
        
        return 1/(np.sqrt(2*np.pi)*s)*np.exp(-1/(2*s**2)*(w-1/x)**2)
    
    def normalization(f,par1,w,s,p,r_mode,par2): 
        """Returns the normalization factor of the function f.
    
        Computes the normalization factor as the summation of the percentile of r_mode plus the integration of the function from r_mode to par2 using the scipy.integrate.quad function. This way numerical errors in the integrration of thin distributions are avoided.
        
        ===========
        Parameters:
        ===========
           **f**
           : function
               Probability distribution function. 
           **par1**
           : number
               Parameter of the function f in kpc.    
           **w**
           : number
               Parameter of the function f in mas.    
           **s**
           : number
               Parameter of the function f in mas.    
           **p**
           : number
               Unnormalized percentile of r_mode.    
           **r_mode**
           : number
               Distance in kpc.      
           **par_2**
           : number
               Superior bound of the integration in kpc.
        ========
        Returns:
        ========
            **number**
                Normalization factor.
       """
        
        N = quad(f, r_mode, par2, args=(w, s, par1),epsrel = 1e-11, epsabs = 1e-11) # We integrate the required posterior from the mode to infinity (par2)
        
        N = N[0] + p # p is the percentile corresponding to the mode, i.e. the integration of the PDF from r0 to r_mode
        
        return N
    
    
    def percentiles(f,r0,r_mode,w,s,par): # Given a distance (i.e. r_mode), it integrates the PDF from r0 to the given distance and returns the unnormalized percentile
        """Returns the percentile corresponding to a given distance r_mode.
        
        This function integrates the function f with parameters w and s from r0 to r_mode using the scipy.integrate.quad function.
        
        ===========
        Parameters:
        ===========    
            **f** 
            : function
                Probability distribution function.
            **r0**
            : number
                Inferior bound in the integration.
            **r_mode**
            : number
                Superio bound in the integration.
            **w** 
            : number
                Parameter of f. 
            **s**
            : number
                Parameter of f.
        ========
        Returns:
        ========
            **number**
                Unnormalized percentile of r_mode.
        """
        p = quad(f,r0,r_mode,args=(w,s,par),epsrel = 1e-12, epsabs = 1e-12)
        return p[0]

    x = np.arange(r0,r_lim,r_lim/10000)
    y_unnorm = [posterior(i,w,s,r_lim) for i in x]
    r_mode=x[y_unnorm.index(max(y_unnorm))]
    p_tot = percentiles(posterior, r0, r_mode, w, s,r_lim)  # Computing the percentile that corresponds to the mode of the PDF
    n_tot = normalization(posterior, r_lim, w, s, p_tot, r_mode,r_lim)  # Computing the normalization constant of the PDF
    y_norm = [posterior(i,w,s,r_lim)/n_tot for i in x]
    posterior_func = interp1d(x,y_norm,bounds_error=False,fill_value=0.0)
    return r_mode, posterior_func, x


def atri_CIcalc(tau, atri_MAP, atri_pdf, prefix='dr?'):
    def atrilofunc(x):
        if (x < atri_MAP) & (x > 0.1):
            return atri_pdf(x) - tau
        else:
            return 1000.0
    
    def atrihifunc(x):
        if (x > atri_MAP) & (x < 4*atri_MAP):
            return atri_pdf(x) - tau
        else:
            return 1000.0

    lolim = fsolve(atrilofunc,atri_MAP*0.9)
    uplim = fsolve(atrihifunc,atri_MAP*1.1)
    ci = quad(atri_pdf,lolim, uplim, epsabs=1e-5)[0]*100
    print('--------------------')
    print(f'Atri distance PDF:')
    print(f'Gaia {prefix}')
    print(f'Most likely distance (kpc): {atri_MAP:.3}')
    print(f'PDF mode: {atri_pdf(atri_MAP):.3}')
    print(f'Limits on distance (kpc): {lolim[0]:.3}--{uplim[0]:.3}')
    print(f'Confidence interval(%): {ci:.3}')
    print(f'Dinstance: {atri_MAP:.3}(-{atri_MAP-lolim[0]:.3}/+{uplim[0]-atri_MAP:.3}) kpc')
    print('--------------------')
    return lolim, uplim, ci

def target_resolver(name,verbose=True):
    """
    A simple SIMBAD name resolver to return coordinates
    """
    simbad_table = Simbad.query_object(name)
    if verbose:
        simbad_table.pprint(show_unit=True)
    coords = SkyCoord(simbad_table['RA'].data[0],simbad_table['DEC'].data[0],
                      unit = (u.hourangle, u.deg),
                      frame = 'icrs')
    return coords

def gaia_search(coords, radius):
    """
    Gaia ADQL query cone searcher
    Returns tables from both DR2 and EDR3
    """
    radius_deg = (radius*u.arcsec).to(u.deg).value
    gaia = TapPlus(url="https://gea.esac.esa.int/tap-server/tap",verbose=False)

    adql_query_dr2 = f"""SELECT *
                         FROM gaiaedr3.gaia_source
                         WHERE 
                             1=CONTAINS(POINT('ICRS', ra, dec),
                                        CIRCLE('ICRS', {coords.ra.deg}, {coords.dec.deg}, {radius_deg}))
                        """

    job_dr2 = gaia.launch_job(adql_query_dr2)
    #print(job_dr2)
    gaia_dr2_result = job_dr2.get_results()
    print(f'DR2: Found {len(gaia_dr2_result)} sources')

    
    adql_query_edr3 = f"""SELECT *
                          FROM gaiaedr3.gaia_source
                          WHERE 
                              1=CONTAINS(POINT('ICRS', ra, dec),
                                         CIRCLE('ICRS', {coords.ra.deg}, {coords.dec.deg}, {radius_deg}))
                        """

    job_edr3 = gaia.launch_job(adql_query_edr3)
    #print(job_edr3)
    gaia_edr3_result = job_edr3.get_results()
    print(f'DR3: Found {len(gaia_edr3_result)} sources')
    return gaia_dr2_result, gaia_edr3_result

def gaia_zpcorr(gaia_table,drno):
    """
    Parallax zero-point correction applier.
    
    For DR2, this is from Chan and Bovy
    
    For EDR3, it is from Lindegren et al. 2020
    
    """
    if drno == 'dr2':
        zpvals = -0.048
    elif drno == 'dr3':
        gmag = gaia_table['phot_g_mean_mag']
        nueffused = gaia_table['nu_eff_used_in_astrometry']
        psc = gaia_table['pseudocolour']
        ecl_lat = gaia_table['ecl_lat']
        soltype = gaia_table['astrometric_params_solved']
        zpvals = zpt.get_zpt(gmag, nueffused, psc, ecl_lat, soltype)
    newcol1 = Column(data=zpvals,name='zpcorr_val')
    newcol2 = Column(data=gaia_table['parallax']-zpvals,name='parallax_zpcorr')
    gaia_table.add_columns([newcol1,newcol2],indexes=[12,12])
    return gaia_table


def comparison(name, dr2, dr3, mode,
               MAP_dr2, pdf_dr2, distrange_dr2,
               MAP_dr3, pdf_dr3, distrange_dr3,
               lolim_dr2, uplim_dr2,
               lolim_dr3, uplim_dr3,
               distxrange = [0.1,10],
               keys = ['DESIGNATION','parallax','parallax_error','parallax_over_error','zpcorr_val','parallax_zpcorr','pmra','pmra_error','pmdec','pmdec_error']):
    
    c = vstack([dr2[keys],dr3[keys]],metadata_conflicts='silent')

    fig = plt.figure(figsize=(14,6))
    ax0 = plt.subplot2grid((3,2),(0,1),fig=fig)
    ax0.set_title(name, fontsize=20)
    ax0.errorbar(x = c[0]['parallax'], xerr = c[0]['parallax_error'], y = ['DR2'], 
                 fmt='x',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3,label='w/o ZP corr')
    ax0.errorbar(x = c[1]['parallax'], xerr = c[1]['parallax_error'], y = ['DR3'], 
                 fmt='x',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax0.errorbar(x = c[0]['parallax_zpcorr'], xerr = c[0]['parallax_error'],y = 0.1,
                 fmt='.',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3,label='w/ ZP corr')
    ax0.errorbar(x = c[1]['parallax_zpcorr'], xerr = c[1]['parallax_error'],y = 1.1,
                 fmt='.',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax0.set_xlabel('Parallax (mas)',fontsize=16)
    ax0.set_ylim(-0.5,1.5)
    ax0.legend(fontsize=16,bbox_to_anchor=(1,1))
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.tick_params(axis='both', which='major', length=5)
    ax0.tick_params(axis='both', which='minor', length=2.5)
    ax0.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    
    ax1 = plt.subplot2grid((3,2),(1,1),fig=fig)
    ax1.errorbar(x = c[0]['pmra'], xerr = c[0]['pmra_error'], y = ['DR2'], 
                 fmt='.',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3)
    ax1.errorbar(x = c[1]['pmra'], xerr = c[1]['pmra_error'], y = ['DR3'], 
                 fmt='.',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax1.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)',fontsize=16)
    ax1.set_ylim(-0.5,1.5)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='major', length=5)
    ax1.tick_params(axis='both', which='minor', length=2.5)
    ax1.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    ax2 = plt.subplot2grid((3,2),(2,1),fig=fig)
    ax2.errorbar(x = c[0]['pmdec'], xerr = c[0]['pmdec_error'], y = ['DR2'], 
                 fmt='.',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3)
    ax2.errorbar(x = c[1]['pmdec'], xerr = c[1]['pmdec_error'], y = ['DR3'], 
                 fmt='.',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax2.set_xlabel(r'$\mu_\delta$ (mas yr$^{-1}$)',fontsize=16)
    ax2.set_ylim(-0.5,1.5)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', length=5)
    ax2.tick_params(axis='both', which='minor', length=2.5)
    ax2.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    ax3 = plt.subplot2grid((3,2),(0,0),fig=fig, rowspan=3)
    if mode == 'Atri':
        ax3.plot(distrange_dr2,pdf_dr2(distrange_dr2),color=colorset[4],label='DR2')
        ax3.plot(distrange_dr3,pdf_dr3(distrange_dr3),color=colorset[9],label='EDR3')
    elif mode == 'BJ':
        ax3.plot(distrange_dr2,pdf_dr2,color=colorset[4],label='DR2')
        ax3.plot(distrange_dr3,pdf_dr3,color=colorset[9],label='EDR3')
    ax3.axvline(MAP_dr2,color=colorset[4],linestyle='--')
    ax3.axvline(lolim_dr2,color=colorset[4],linestyle=':')
    ax3.axvline(uplim_dr2,color=colorset[4],linestyle=':')
    ax3.axvline(MAP_dr3,color=colorset[9],linestyle='--')
    ax3.axvline(lolim_dr3,color=colorset[9],linestyle=':')
    ax3.axvline(uplim_dr3,color=colorset[9],linestyle=':')
    ax3.legend(loc=1, fontsize=14)
    ax3.set_xlim(distxrange)
    ax3.set_ylim(0,)
    if mode == 'Atri':
        ax3.set_title(f'DR2: $d$={MAP_dr2:.3}(-{MAP_dr2-lolim_dr2[0]:.3}/+{uplim_dr2[0]-MAP_dr2:.3}) kpc\nEDR3: $d$={MAP_dr3:.3}(-{MAP_dr3-lolim_dr3[0]:.3}/+{uplim_dr3[0]-MAP_dr3:.3}) kpc', fontsize=20)
    elif mode == 'BJ':
        ax3.set_title(f'DR2: $d$={MAP_dr2:.3}(-{MAP_dr2-lolim_dr2:.3}/+{uplim_dr2-MAP_dr2:.3}) kpc\nEDR3: $d$={MAP_dr3:.3}(-{MAP_dr3-lolim_dr3:.3}/+{uplim_dr3-MAP_dr3:.3}) kpc', fontsize=20)
    ax3.set_xlabel(f'Distance (kpc; {mode} prior)',fontsize=16)
    ax3.set_ylabel(r'PDF (kpc$^{-1}$)',fontsize=16)
    ax3.minorticks_on()
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.tick_params(axis='both', which='major', length=9)
    ax3.tick_params(axis='both', which='minor', length=4.5)
    ax3.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    fig.subplots_adjust(hspace=0.6, wspace=0.11)
    return c, fig


def comparison_all(name, dr2, dr3, 
                   atri_MAP_dr2, atri_pdf_dr2, distrange_dr2,
                   atri_MAP_dr3, atri_pdf_dr3, distrange_dr3,
                   atri_lolim_dr2, atri_uplim_dr2,
                   atri_lolim_dr3, atri_uplim_dr3,
                   bj_MAP_dr2, bj_pdf_dr2, bj_distrange_dr2,
                   bj_MAP_dr3, bj_pdf_dr3, bj_distrange_dr3,
                   bj_lolim_dr2, bj_uplim_dr2,
                   bj_lolim_dr3, bj_uplim_dr3,
                   distxrange = [0.1,10],
                   keys = ['DESIGNATION','parallax','parallax_error','parallax_over_error','zpcorr_val','parallax_zpcorr','pmra','pmra_error','pmdec','pmdec_error']):
    
    c = vstack([dr2[keys],dr3[keys]],metadata_conflicts='silent')

    fig = plt.figure(figsize=(14,6))
    ax0 = plt.subplot2grid((3,2),(0,1),fig=fig)
    ax0.set_title(name, fontsize=20)
    #ax0.errorbar(x = c[0]['parallax'], xerr = c[0]['parallax_error'], y = ['DR2'], 
    #             fmt='x',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3,label='w/o ZP corr')
    ax0.errorbar(x = c[1]['parallax'], xerr = c[1]['parallax_error'], y = ['DR3'], 
                 fmt='x',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    #ax0.errorbar(x = c[0]['parallax_zpcorr'], xerr = c[0]['parallax_error'],y = 0.1,
    #             fmt='.',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3,label='w/ ZP corr')
    ax0.errorbar(x = c[1]['parallax_zpcorr'], xerr = c[1]['parallax_error'],y = 1.1,
                 fmt='.',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax0.set_xlabel('Parallax (mas)',fontsize=16)
    ax0.set_ylim(-0.5,1.5)
    #ax0.legend(fontsize=16,bbox_to_anchor=(1,1))
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.tick_params(axis='both', which='major', length=5)
    ax0.tick_params(axis='both', which='minor', length=2.5)
    ax0.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    
    ax1 = plt.subplot2grid((3,2),(1,1),fig=fig)
    #ax1.errorbar(x = c[0]['pmra'], xerr = c[0]['pmra_error'], y = ['DR2'], 
    #             fmt='.',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3)
    ax1.errorbar(x = c[1]['pmra'], xerr = c[1]['pmra_error'], y = ['DR3'], 
                 fmt='.',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax1.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)',fontsize=16)
    ax1.set_ylim(-0.5,1.5)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='major', length=5)
    ax1.tick_params(axis='both', which='minor', length=2.5)
    ax1.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    ax2 = plt.subplot2grid((3,2),(2,1),fig=fig)
    #ax2.errorbar(x = c[0]['pmdec'], xerr = c[0]['pmdec_error'], y = ['DR2'], 
    #             fmt='.',ms = 10,color=colorset[4],elinewidth=1.9,capsize=3)
    ax2.errorbar(x = c[1]['pmdec'], xerr = c[1]['pmdec_error'], y = ['DR3'], 
                 fmt='.',ms = 10,color=colorset[9],elinewidth=1.9,capsize=3)
    ax2.set_xlabel(r'$\mu_\delta$ (mas yr$^{-1}$)',fontsize=16)
    ax2.set_ylim(-0.5,1.5)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', length=5)
    ax2.tick_params(axis='both', which='minor', length=2.5)
    ax2.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    ax3 = plt.subplot2grid((3,2),(0,0),fig=fig, rowspan=3)
    #ax3.plot(distrange_dr2,atri_pdf_dr2(distrange_dr2),color=colorset[4],label='DR2 (Atri)')
    #ax3.plot(bj_distrange_dr2,bj_pdf_dr2,'--',color=colorset[4],label='DR2 (Bailer-Jones)',lw=3,alpha=0.2)
    ax3.plot(distrange_dr3,atri_pdf_dr3(distrange_dr3),color=colorset[9],label='EDR3 (Atri)')
    ax3.plot(bj_distrange_dr3,bj_pdf_dr3,'--',color=colorset[9],label='EDR3 (Bailer-Jones)',lw=3,alpha=0.2)
    ax3.legend(loc=1,fontsize=16)
    ax3.set_xlim(distxrange)
    ax3.set_ylim(0,)
    ax3.set_title(f'EDR3: $d$(Atri)={atri_MAP_dr3:.2}(-{atri_MAP_dr3-atri_lolim_dr3[0]:.2}/+{atri_uplim_dr3[0]-atri_MAP_dr3:.2}) kpc' + 
                  f'; $d$(BJ)={bj_MAP_dr3:.2}(-{bj_MAP_dr3-bj_lolim_dr3:.2}/+{bj_uplim_dr3-bj_MAP_dr3:.2}) kpc', fontsize=16)
    ax3.set_xlabel(r'Distance (kpc)',fontsize=16)
    ax3.set_ylabel(r'PDF (kpc$^{-1}$)',fontsize=16)
    ax3.minorticks_on()
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.tick_params(axis='both', which='major', length=9)
    ax3.tick_params(axis='both', which='minor', length=4.5)
    ax3.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    fig.subplots_adjust(hspace=0.6, wspace=0.11)
    return c, fig

def gaiaedr3_plots(src_name, search_rad, tau_dr2, tau_dr3, distance_plotting_range, distance_method):
    source_name = src_name
    gaia_tab_dr2, gaia_tab_dr3 = gaia_search(target_resolver(source_name), search_rad)
    try:
        gaia_zpcorr(gaia_tab_dr2,'dr2');
        gaia_zpcorr(gaia_tab_dr3,'dr3');
    except:
        if len(gaia_tab_dr2) != 1 or len(gaia_tab_dr3) != 1:
            print('ERROR: number of Gaia counterparts found != 1. Change the search radius.')
        else:
            print('ERROR: ZP correction failed. The source may not have 5-parameter solution.')
        keys = ['DESIGNATION','ra','dec','parallax','parallax_error','parallax_over_error','pmra','pmra_error','pmdec','pmdec_error']
        return vstack([gaia_tab_dr2[keys],gaia_tab_dr3[keys]],metadata_conflicts='silent'), None

    print('Separation between SIMBAD coordinates and the Gaia counterpart (DR2):',round(target_resolver(source_name,verbose=False).separation(SkyCoord(gaia_tab_dr2['ra'],gaia_tab_dr2['dec']))[0].arcsec,3),'arcsec')
    print('Separation between SIMBAD coordinates and the Gaia counterpart (EDR3):',round(target_resolver(source_name,verbose=False).separation(SkyCoord(gaia_tab_dr3['ra'],gaia_tab_dr3['dec']))[0].arcsec,3),'arcsec')
    
    if distance_method == 'both':
        atri_MAP_dr2, atri_pdf_dr2, distrange_dr2 = atri_dist(gaia_tab_dr2['parallax_zpcorr'][0],
                                                              gaia_tab_dr2['parallax_error'][0],
                                                              gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0])

        atri_MAP_dr3, atri_pdf_dr3, distrange_dr3 = atri_dist(gaia_tab_dr3['parallax_zpcorr'][0],
                                                              gaia_tab_dr3['parallax_error'][0],
                                                              gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0])


        atri_lolim_dr2, atri_uplim_dr2, ci_dr2 = atri_CIcalc(tau_dr2,atri_MAP_dr2, atri_pdf_dr2,'dr2')
        atri_lolim_dr3, atri_uplim_dr3, ci_dr3 = atri_CIcalc(tau_dr3,atri_MAP_dr3, atri_pdf_dr3,'dr3')

        bj_pdf_dr2 ,bj_pointest_dr2 = bailerjones(gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0],
                                                  gaia_tab_dr2['parallax_zpcorr'][0],
                                                  gaia_tab_dr2['parallax_error'][0])

        bj_pdf_dr3 ,bj_pointest_dr3 = bailerjones(gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0],
                                                  gaia_tab_dr3['parallax_zpcorr'][0],
                                                  gaia_tab_dr3['parallax_error'][0])


        tab, fig = comparison_all(source_name,gaia_tab_dr2,gaia_tab_dr3,
                                  atri_MAP_dr2, atri_pdf_dr2, distrange_dr2,
                                  atri_MAP_dr3, atri_pdf_dr3, distrange_dr3,
                                  atri_lolim_dr2, atri_uplim_dr2,
                                  atri_lolim_dr3, atri_uplim_dr3, 
                                  bj_pointest_dr2['point_est'][0],bj_pdf_dr2['pdf'].data,bj_pdf_dr2['d'].data,
                                  bj_pointest_dr3['point_est'][0],bj_pdf_dr3['pdf'].data,bj_pdf_dr3['d'].data,
                                  bj_pointest_dr2['point_est'][1],bj_pointest_dr2['point_est'][2],
                                  bj_pointest_dr3['point_est'][1],bj_pointest_dr3['point_est'][2],
                                  distxrange=distance_plotting_range);
    elif distance_method == 'atri':
        atri_MAP_dr2, atri_pdf_dr2, distrange_dr2 = atri_dist(gaia_tab_dr2['parallax_zpcorr'][0],
                                                              gaia_tab_dr2['parallax_error'][0],
                                                              gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0])

        atri_MAP_dr3, atri_pdf_dr3, distrange_dr3 = atri_dist(gaia_tab_dr3['parallax_zpcorr'][0],
                                                              gaia_tab_dr3['parallax_error'][0],
                                                              gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0])


        atri_lolim_dr2, atri_uplim_dr2, ci_dr2 = atri_CIcalc(tau_dr2,atri_MAP_dr2, atri_pdf_dr2,'dr2')
        atri_lolim_dr3, atri_uplim_dr3, ci_dr3 = atri_CIcalc(tau_dr3,atri_MAP_dr3, atri_pdf_dr3,'dr3')

        tab, fig = comparison(source_name,gaia_tab_dr2,gaia_tab_dr3, 'Atri',
                              atri_MAP_dr2, atri_pdf_dr2, distrange_dr2,
                              atri_MAP_dr3, atri_pdf_dr3, distrange_dr3,
                              atri_lolim_dr2, atri_uplim_dr2,
                              atri_lolim_dr3, atri_uplim_dr3,
                              distxrange=distance_plotting_range);
    elif distance_method == 'bj':
        bj_pdf_dr2 ,bj_pointest_dr2 = bailerjones(gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0],
                                                  gaia_tab_dr2['parallax_zpcorr'][0],
                                                  gaia_tab_dr2['parallax_error'][0])

        bj_pdf_dr3 ,bj_pointest_dr3 = bailerjones(gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0],
                                                  gaia_tab_dr3['parallax_zpcorr'][0],
                                                  gaia_tab_dr3['parallax_error'][0])

        tab, fig = comparison(source_name,gaia_tab_dr2,gaia_tab_dr3, 'BJ',
                              bj_pointest_dr2['point_est'][0],bj_pdf_dr2['pdf'].data,bj_pdf_dr2['d'].data,
                              bj_pointest_dr3['point_est'][0],bj_pdf_dr3['pdf'].data,bj_pdf_dr3['d'].data,
                              bj_pointest_dr2['point_est'][1],bj_pointest_dr2['point_est'][2],
                              bj_pointest_dr3['point_est'][1],bj_pointest_dr3['point_est'][2],
                              distxrange=distance_plotting_range);

    return tab, fig


def gaiaedr3_plots_coords(ra, dec, search_rad, tau_dr2, tau_dr3, distance_plotting_range, distance_method):
    source_name = ra+'\n'+dec
    gaia_tab_dr2, gaia_tab_dr3 = gaia_search(SkyCoord(ra,dec,unit = (u.hourangle, u.deg), frame = 'icrs'), search_rad)
    try:
        gaia_zpcorr(gaia_tab_dr2,'dr2');
        gaia_zpcorr(gaia_tab_dr3,'dr3');
    except:
        if len(gaia_tab_dr2) != 1 or len(gaia_tab_dr3) != 1:
            print('ERROR: number of Gaia counterparts found != 1. Change the search radius.')
        else:
            print('ERROR: ZP correction failed. The source may not have 5-parameter solution.')
        keys = ['DESIGNATION','ra','dec','parallax','parallax_error','parallax_over_error','pmra','pmra_error','pmdec','pmdec_error']
        return vstack([gaia_tab_dr2[keys],gaia_tab_dr3[keys]],metadata_conflicts='silent'), None

    print('Separation between INPUT coordinates and the Gaia counterpart (DR2):',round(SkyCoord(ra,dec,unit = (u.hourangle, u.deg)).separation(SkyCoord(gaia_tab_dr2['ra'],gaia_tab_dr2['dec']))[0].arcsec,3),'arcsec')
    print('Separation between INPUT coordinates and the Gaia counterpart (EDR3):',round(SkyCoord(ra,dec,unit = (u.hourangle, u.deg)).separation(SkyCoord(gaia_tab_dr3['ra'],gaia_tab_dr3['dec']))[0].arcsec,3),'arcsec')

    if distance_method == 'both':
        atri_MAP_dr2, atri_pdf_dr2, distrange_dr2 = atri_dist(gaia_tab_dr2['parallax_zpcorr'][0],
                                                              gaia_tab_dr2['parallax_error'][0],
                                                              gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0])

        atri_MAP_dr3, atri_pdf_dr3, distrange_dr3 = atri_dist(gaia_tab_dr3['parallax_zpcorr'][0],
                                                              gaia_tab_dr3['parallax_error'][0],
                                                              gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0])


        atri_lolim_dr2, atri_uplim_dr2, ci_dr2 = atri_CIcalc(tau_dr2,atri_MAP_dr2, atri_pdf_dr2,'dr2')
        atri_lolim_dr3, atri_uplim_dr3, ci_dr3 = atri_CIcalc(tau_dr3,atri_MAP_dr3, atri_pdf_dr3,'dr3')

        bj_pdf_dr2 ,bj_pointest_dr2 = bailerjones(gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0],
                                                  gaia_tab_dr2['parallax_zpcorr'][0],
                                                  gaia_tab_dr2['parallax_error'][0])

        bj_pdf_dr3 ,bj_pointest_dr3 = bailerjones(gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0],
                                                  gaia_tab_dr3['parallax_zpcorr'][0],
                                                  gaia_tab_dr3['parallax_error'][0])


        tab, fig = comparison_all(source_name,gaia_tab_dr2,gaia_tab_dr3,
                                  atri_MAP_dr2, atri_pdf_dr2, distrange_dr2,
                                  atri_MAP_dr3, atri_pdf_dr3, distrange_dr3,
                                  atri_lolim_dr2, atri_uplim_dr2,
                                  atri_lolim_dr3, atri_uplim_dr3, 
                                  bj_pointest_dr2['point_est'][0],bj_pdf_dr2['pdf'].data,bj_pdf_dr2['d'].data,
                                  bj_pointest_dr3['point_est'][0],bj_pdf_dr3['pdf'].data,bj_pdf_dr3['d'].data,
                                  bj_pointest_dr2['point_est'][1],bj_pointest_dr2['point_est'][2],
                                  bj_pointest_dr3['point_est'][1],bj_pointest_dr3['point_est'][2],
                                  distxrange=distance_plotting_range);
    elif distance_method == 'atri':
        atri_MAP_dr2, atri_pdf_dr2, distrange_dr2 = atri_dist(gaia_tab_dr2['parallax_zpcorr'][0],
                                                              gaia_tab_dr2['parallax_error'][0],
                                                              gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0])

        atri_MAP_dr3, atri_pdf_dr3, distrange_dr3 = atri_dist(gaia_tab_dr3['parallax_zpcorr'][0],
                                                              gaia_tab_dr3['parallax_error'][0],
                                                              gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0])


        atri_lolim_dr2, atri_uplim_dr2, ci_dr2 = atri_CIcalc(tau_dr2,atri_MAP_dr2, atri_pdf_dr2,'dr2')
        atri_lolim_dr3, atri_uplim_dr3, ci_dr3 = atri_CIcalc(tau_dr3,atri_MAP_dr3, atri_pdf_dr3,'dr3')

        tab, fig = comparison(source_name,gaia_tab_dr2,gaia_tab_dr3, 'Atri',
                              atri_MAP_dr2, atri_pdf_dr2, distrange_dr2,
                              atri_MAP_dr3, atri_pdf_dr3, distrange_dr3,
                              atri_lolim_dr2, atri_uplim_dr2,
                              atri_lolim_dr3, atri_uplim_dr3,
                              distxrange=distance_plotting_range);
    elif distance_method == 'bj':
        bj_pdf_dr2 ,bj_pointest_dr2 = bailerjones(gaia_tab_dr2['ra'][0],gaia_tab_dr2['dec'][0],
                                                  gaia_tab_dr2['parallax_zpcorr'][0],
                                                  gaia_tab_dr2['parallax_error'][0])

        bj_pdf_dr3 ,bj_pointest_dr3 = bailerjones(gaia_tab_dr3['ra'][0],gaia_tab_dr3['dec'][0],
                                                  gaia_tab_dr3['parallax_zpcorr'][0],
                                                  gaia_tab_dr3['parallax_error'][0])

        tab, fig = comparison(source_name,gaia_tab_dr2,gaia_tab_dr3, 'BJ',
                              bj_pointest_dr2['point_est'][0],bj_pdf_dr2['pdf'].data,bj_pdf_dr2['d'].data,
                              bj_pointest_dr3['point_est'][0],bj_pdf_dr3['pdf'].data,bj_pdf_dr3['d'].data,
                              bj_pointest_dr2['point_est'][1],bj_pointest_dr2['point_est'][2],
                              bj_pointest_dr3['point_est'][1],bj_pointest_dr3['point_est'][2],
                              distxrange=distance_plotting_range);

    return tab, fig

def cipe(src_ra, src_dec, counterpart_separation, region_radius=0.1, numpoints=10000):
    counterpart_separation = counterpart_separation * u.arcsec
    region_radius = region_radius * u.degree
    tap_cap = 100000
    tap_server = TapPlus(url='https://gea.esac.esa.int/tap-server/tap',verbose=False)
    catalog = 'gaiaedr3.gaia_source'

    query = "SELECT TOP " + str(tap_cap) + \
            " * FROM " + catalog + " WHERE ra BETWEEN " + \
            str(src_ra.value - region_radius.value) + \
            " AND " + str(src_ra.value + region_radius.value) + \
            " AND dec BETWEEN " + str(src_dec.value - region_radius.value) + \
            " AND " + str(src_dec.value + region_radius.value)

    search = tap_server.launch_job(query)
    results = search.get_results()
    print('Number of Gaia sources:' + str(len(results)))

    if len(results) == tap_cap:
        print('WARNING: Gaia contains too many sources in the region >' + str(tap_cap) + '). Region may be too large.')

    gaia_srclist = SkyCoord(ra=results['ra'], dec=results['dec'])

    fake_srclist = SkyCoord(ra=src_ra + (np.random.rand(numpoints) - 0.5) * 2 * region_radius,
                            dec=src_dec + (np.random.rand(numpoints) - 0.5) * 2 * region_radius)

    sep_dist = fake_srclist.match_to_catalog_sky(gaia_srclist)[1].to(u.arcsec).value

    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.hist(sep_dist, bins=50, color='#034BCA', edgecolor='w', density=True, label='Simulations')
    model_x = np.linspace(0,8,1000)
    params = st.genextreme.fit(sep_dist)
    model_y = st.genextreme(*params).pdf(model_x)
    p_less = len(sep_dist[sep_dist <= counterpart_separation.value])/len(sep_dist)*100
    ax1.plot(model_x,model_y,color='#EB24F4',label='Gumble fit')
    ax1.axvline(0.51,color='k',linestyle='--')
    ax1.set_title(f"$P(d<{counterpart_separation.value}'')={p_less:.3}\%$",fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_xlabel(r'Distance to closest random Gaia source (arcsec)',fontsize=14)
    ax1.set_ylabel(r'Probabilty density (arcsec$^{-1}$)',fontsize=14)
    ax1.set_xlim(0,8)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='both', which='major', length=9)
    ax1.tick_params(axis='both', which='minor', length=4.5)
    ax1.tick_params(axis='both', which='both', direction='in', right=True, top=True)

    fig2 = plt.figure(figsize=(6,4))
    ax = fig2.add_subplot(1,1,1)
    ax.plot(gaia_srclist.ra.deg, gaia_srclist.dec.deg, '*b', label='Gaia')
    ax.plot(fake_srclist.ra.deg, fake_srclist.dec.deg, '.r', label='fake', alpha=0.5, ms=0.7)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='major', length=9)
    ax.tick_params(axis='both', which='minor', length=4.5)
    ax.tick_params(axis='both', which='both', direction='in', right=True, top=True)


    return fig1, fig2

def gaia_src_summary_nb(source_name, search_radius, distance_range, distance_prior, tau_dr2, tau_dr3, 
                        method ='simbad', source_ra='00 00 00.00', source_dec='00 00 00.00'):
    if method == 'simbad':
        tab, fig = gaiaedr3_plots(source_name, search_radius, tau_dr2, tau_dr3, 
                                  distance_range, distance_prior)
    elif method == 'coords':
        tab, fig = gaiaedr3_plots_coords(source_ra, source_dec, search_radius, 
                                         tau_dr2, tau_dr3, distance_range, distance_prior)
    
    lastab = bailerjones_new(tab['DESIGNATION'][1][10:])
    txt_geo = str(round((lastab['r_med_geo'].data)[0]/1000,2))+\
              '(-'+str(round((lastab['r_med_geo'].data - lastab['r_lo_geo'].data)[0]/1000,2))+\
              '/+'+str(round((lastab['r_hi_geo'].data - lastab['r_med_geo'].data)[0]/1000,2))+')'

    txt_geophot = str(round((lastab['r_med_photogeo'].data)[0]/1000,2))+\
                  '(-'+str(round((lastab['r_med_photogeo'].data - lastab['r_lo_photogeo'].data)[0]/1000,2))+\
                  '/+'+str(round((lastab['r_hi_photogeo'].data - lastab['r_med_photogeo'].data)[0]/1000,2))+')'

    lastab.add_column([txt_geo],name='geodist (kpc)')
    lastab.add_column([txt_geophot],name='geophot dist (kpc)')
    print('Table 1: DR2 and EDR3 counterparts')
    display(tab)
    print('Table 2: Bailer-Jones 2020 distances')
    display(lastab)
    return fig

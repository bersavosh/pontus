import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
colorset = ['#000000','#00270C','#00443C','#005083',
            '#034BCA','#483CFC','#9C2BFF','#EB24F4',
            '#FF2DC2','#FF4986','#FF7356','#FFA443',
            '#EBD155','#D3F187','#D7FFC8','#FFFFFF']

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, vstack
from astroquery.utils.tap.core import TapPlus
from astroquery.simbad import Simbad

from zero_point import zpt
zpt.load_tables()

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
    ci = quad(atri_pdf,lolim, uplim)[0]*100
    print(f'Gaia {prefix}')
    print(f'Most likely distance (kpc): {atri_MAP:.3}')
    print(f'PDF mode: {atri_pdf(atri_MAP):.3}')
    print(f'Limits on distance (kpc): {lolim[0]:.3}--{uplim[0]:.3}')
    print(f'Confidence interval(%): {ci:.3}')
    print(f'Dinstance: {atri_MAP:.3}(-{atri_MAP-lolim[0]:.3}/+{uplim[0]-atri_MAP:.3}) kpc')
    print('--------------------')
    return lolim, uplim, ci

def target_resolver(name):
    """
    A simple SIMBAD name resolver to return coordinates
    """
    simbad_table = Simbad.query_object(name)
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
    gaia = TapPlus(url="https://gea.esac.esa.int/tap-server/tap")

    adql_query_dr2 = f"""SELECT *
                         FROM gaiadr2.gaia_source
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

def comparison(name, dr2, dr3, keys = ['designation','parallax','parallax_error','parallax_over_error','zpcorr_val','parallax_zpcorr','pmra','pmra_error','pmdec','pmdec_error']):
    c = vstack([dr2[keys],dr3[keys]])

    fig = plt.figure(figsize=(8,6))
    ax0 = fig.add_subplot(3,1,1)
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
    
    
    ax1 = fig.add_subplot(3,1,2)
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
    
    
    ax2 = fig.add_subplot(3,1,3)
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
    fig.subplots_adjust(hspace=0.6)
    return fig

def comparison(name, dr2, dr3, 
               atri_MAP_dr2, atri_pdf_dr2, distrange_dr2,
               atri_MAP_dr3, atri_pdf_dr3, distrange_dr3,
               lolim_dr2, uplim_dr2, ci_dr2,
               lolim_dr3, uplim_dr3, ci_dr3,
               distxrange = [0.1,10],
               keys = ['designation','parallax','parallax_error','parallax_over_error','zpcorr_val','parallax_zpcorr','pmra','pmra_error','pmdec','pmdec_error']):
    
    c = vstack([dr2[keys],dr3[keys]])

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
    ax3.plot(distrange_dr2,atri_pdf_dr2(distrange_dr2),color=colorset[4],label='DR2')
    ax3.axvline(atri_MAP_dr2,color=colorset[4],linestyle='--')
    ax3.axvline(lolim_dr2,color=colorset[4],linestyle=':')
    ax3.axvline(uplim_dr2,color=colorset[4],linestyle=':')
    ax3.plot(distrange_dr3,atri_pdf_dr3(distrange_dr3),color=colorset[9],label='EDR3')
    ax3.axvline(atri_MAP_dr3,color=colorset[9],linestyle='--')
    ax3.axvline(lolim_dr3,color=colorset[9],linestyle=':')
    ax3.axvline(uplim_dr3,color=colorset[9],linestyle=':')
    ax3.legend(fontsize=16)
    ax3.set_xlim(distxrange)
    ax3.set_ylim(0,)
    ax3.set_title(f'DR2: $d$={atri_MAP_dr2:.3}(-{atri_MAP_dr2-lolim_dr2[0]:.3}/+{uplim_dr2[0]-atri_MAP_dr2:.3}) kpc\nEDR3: $d$={atri_MAP_dr3:.3}(-{atri_MAP_dr3-lolim_dr3[0]:.3}/+{uplim_dr3[0]-atri_MAP_dr3:.3}) kpc', fontsize=20)
    ax3.set_xlabel(r'Distance (kpc)',fontsize=16)
    ax3.set_ylabel(r'PDF (kpc$^{-1}$)',fontsize=16)
    ax3.minorticks_on()
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.tick_params(axis='both', which='major', length=9)
    ax3.tick_params(axis='both', which='minor', length=4.5)
    ax3.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    
    fig.subplots_adjust(hspace=0.6, wspace=0.11)
    return c, fig

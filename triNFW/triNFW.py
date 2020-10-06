#!/usr/bin/env python

#############################################
#
# Project NFW
#
#############################################


import numpy as np
from math import *
import cosmolopy.distance as cosdist
import cosmolopy.density as cosdens
import cosmolopy.constants as cosconst
import warnings

# ---
# Set a default cosmology
# ---
cosmo           =       {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
cosmo           =       cosdist.set_omega_k_0(cosmo)

# ---
# Define the vir_overden
# ---
def calc_vir_overden(zd, cosmo = cosmo):
    """
    This is the overdensity wrt to mean density of the Universe at redshift zd when the halo is viriliazed.
    This is a fitting formula given in equation C19 in Nakamura and Suto (1997).
    Parameters:
        -`zd`: float. The halo redshift.
        -`cosmo`: dict. The cosmology parameter for this halo. It has to be compatible with
    the format of the cosmolopy module.
    """
    return 18.0*pi*pi*( 1.0 + 0.4093 * (cosdens.omega_M_z(zd,**cosmo)**-1 - 1.0)**0.9052 )


#############################################
#
# Module
#
#############################################

class pNFW(object):
    """
    This is the class for projected NFW (hereafter pNFW).
    There are several input parameters:

    Parameters:
        -`mass`:
        -`concen`:
        -`overden`:
        -`wrt`:
        -`zd`:
        -`qa`: the axis ratio, 0 < aq := a/c <= qb := b/c <= 1.
        -`qb`: the axis ratio, 0 < aq := a/c <= qb := b/c <= 1.
        -`theta`:
        -`phi`:
        -`psi`:
        -`cosmo`
    """
    def __init__(self,
                 mass    = 3E14,
                 concen  = 3.0,
                 zd      = 0.3,
                 overden = 200,
                 wrt     = "crit",
                 qa      = 1.0,
                 qb      = 1.0,
                 theta   = 0.0,
                 phi     = 0.0,
                 psi     = 0.0,
                 cosmo   = cosmo):
        # sanity check
        if  not ( (zd >  0.0) and (mass > 0.0) and (concen > 0.0) ):
            raise ValueError("The input halo params are wrong, (zd, mass, concen):", zd, mass, concen, ".")
        if  wrt not in ["crit", "mean"]:
            raise NameError("The input wrt", wrt, "has to be crit or mean.")
        if  not  0 < qa <= qb <= 1.0:
            raise ValueError("The input axis ratios, qa, qb, are wrong, (qa, qb):", qa, qb, ".")
        # initiate
        self.zd         =       float(zd)
        self.mass       =       float(mass)
        self.concen     =       float(concen)
        self.overden    =       float(overden)
        self.wrt        =       str(wrt)
        self.cosmo      =       cosdist.set_omega_k_0(cosmo)                           # set cosmology
        self.vir_overden=       calc_vir_overden(zd = self.zd, cosmo = self.cosmo)     # calculate virialized overden
        self.qa         =       float(qa)
        self.qb         =       float(qb)
        self.theta      =       float(theta)
        self.phi        =       float(phi)
        self.psi        =       float(psi)
        # cosmology properties
        self.da         =       cosdist.angular_diameter_distance(self.zd, **self.cosmo)
        self.ez         =       cosdist.e_z(self.zd, **self.cosmo)
        self.rho_crit   =       cosdens.cosmo_densities(**self.cosmo)[0] * self.ez**2    # Msun/Mpc3
        self.rho_mean   =       self.rho_crit * cosdens.omega_M_z(self.zd, **self.cosmo) # Msun/Mpc3
        self.arcmin2mpc =       1.0 / 60.0 * pi / 180.0  * self.da                       # Mpc/arcmin
        # set up the compatibility between overden and wrt
        if     self.overden  <  0.0:  self.wrt    =       "vir"
        # set up halo radius / rhos[Msun/Mpc^3] / rs[Mpc]
        if   self.wrt   ==      "mean":
            self.radmpc           =   ( self.mass / (4.0 * pi / 3.0 * self.qa * self.qb * self.overden * self.rho_mean) )**(1.0/3.0)
            self.radarcmin        =   self.radmpc / self.arcmin2mpc
            self._factorO         =   self.overden * self.rho_mean
            self.rhos             =   self.overden * self.rho_mean * (self.concen**3 / 3.0) / ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) )
            self.rs               =   self.radmpc / self.concen

        elif self.wrt   ==      "crit":
            self.radmpc           =   ( self.mass / (4.0 * pi / 3.0 * self.qa * self.qb * self.overden * self.rho_crit) )**(1.0/3.0)
            self.radarcmin        =   self.radmpc / self.arcmin2mpc
            self._factorO         =   self.overden * self.rho_crit
            self.rhos             =   self.overden * self.rho_crit * (self.concen**3 / 3.0) / ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) )
            self.rs               =   self.radmpc / self.concen

        elif self.wrt   ==      "vir":
            self.radmpc           =   ( self.mass / (4.0 * pi / 3.0 * self.qa * self.qb * self.vir_overden * self.rho_mean) )**(1.0/3.0)
            self.radarcmin        =   self.radmpc / self.arcmin2mpc
            self._factorO         =   self.vir_overden * self.rho_mean
            self.rhos             =   self.vir_overden * self.rho_mean * (self.concen**3 / 3.0) / ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) )
            self.rs               =   self.radmpc / self.concen
        # coordinate transformation - it follows Umetsu+15
        self.jj         =       np.cos(self.theta)**2 * ( np.cos(self.phi)**2 / self.qa**2 + np.sin(self.phi)**2 / self.qb**2 ) + \
                                np.sin(self.theta)**2 / ( self.qa**2 * self.qb**2 )
        self.kk         =       np.sin(self.phi) * np.cos(self.phi) * np.cos(self.theta) * (1.0/self.qa**2 - 1.0/self.qb**2)
        self.ll         =       np.sin(self.phi)**2 / self.qa**2 + np.cos(self.phi)**2 / self.qb**2
        self.ff         =       np.sin(self.theta)**2 * (np.sin(self.phi)**2 / self.qa**2 + np.cos(self.phi)**2 / self.qb**2) + np.cos(self.theta)**2
        self.q_x        =       np.sqrt( 2 * self.ff / (self.jj + self.ll - np.sqrt((self.jj - self.ll)**2 + 4 * self.kk**2)) )
        self.q_y        =       np.sqrt( 2 * self.ff / (self.jj + self.ll + np.sqrt((self.jj - self.ll)**2 + 4 * self.kk**2)) )
        self.q_proj     =       self.q_y / self.q_x
        self.e_para     =       np.sqrt( self.q_proj / (self.qa * self.qb) ) / self.ff**(3.0/4.0)
        self.fgeo       =       self.e_para / np.sqrt(self.q_proj)
        self.sigma_s    =       2 * self.rhos * self.rs / np.sqrt( self.ff )
        try:
            self.triaxiality=       (1.0 - self.qb**2.0)/(1.0 - self.qa**2.0)
        except ZeroDivisionError:
            self.triaxiality=       0.0

        # derive delta_psi, which is the positional angle of the projected major axis on the plan of sky with respect to the coordinate
        # BEFORE the third rotation (psi) applied.
        if      self.kk     ==  0.0:
            if   self.jj >= self.ll:
                self.delta_psi    =   - pi / 2.0
            else:
                self.delta_psi    =     0.0
        elif    self.kk     >   0.0:
            if   self.jj == self.ll:
                self.delta_psi    =   0.5 * np.arctan( np.inf )
            elif self.jj >  self.ll:
                self.delta_psi    =   0.5 * np.arctan( 2 * self.kk / (self.jj - self.ll) ) - pi / 2.0
            else:
                self.delta_psi    =   0.5 * np.arctan( 2 * self.kk / (self.jj - self.ll) )
        else:
            if   self.jj == self.ll:
                self.delta_psi    =   0.5 * np.arctan(-np.inf )
            elif self.jj >  self.ll:
                self.delta_psi    =   0.5 * np.arctan( 2 * self.kk / (self.jj - self.ll) ) - pi / 2.0
            else:
                self.delta_psi    =   0.5 * np.arctan( 2 * self.kk / (self.jj - self.ll) )
        '''
        if          self.kk     ==  0.0:
            self.psi    =   0.0
        elif       self.jj == self.ll  and  self.kk   >   0.0:
            self.psi    =   0.5 * np.arctan( np.inf )
        elif       self.jj == self.ll  and  self.kk   <   0.0:
            self.psi    =   0.5 * np.arctan(-np.inf )
        else:
            self.psi    =   0.5 * np.arctan( 2 * self.kk / (self.jj - self.ll) )
        '''
        # After the rotation of third Euler angle-- psi
        # More informatin please see the `Sigma_XY` method.
        self.j2prime    =       ( self.jj * np.cos(self.psi)**2.0 + 2.0 * self.kk * np.cos(self.psi) * np.sin(self.psi) + self.ll * np.sin(self.psi)**2.0 )
        self.k2prime    =       ( np.cos(self.psi) * np.sin(self.psi) * (self.ll - self.jj) + self.kk * (np.cos(self.psi)**2.0 - np.sin(self.psi)**2.0)   )
        self.l2prime    =       ( self.jj * np.sin(self.psi)**2.0 - 2.0 * self.kk * np.cos(self.psi) * np.sin(self.psi) + self.ll * np.cos(self.psi)**2.0 )
        # return
        return
    # ---
    # profile
    # ---
    def Sigma(self, zeta):
        """
        Follows eq~36 in Umetsu+15.
        """
        # sanitize
        zeta        =       np.array(zeta, ndmin=1)
        # derive xi
        xi          =       self.q_x * zeta
        # derive xi_s
        xi_s        =       self.q_x * self.rs
        # define f2d
        def f2d(X):
            # sanitize
            X = np.array(X, ndmin=1)
            # case_smaller_than_one
            case_smaller_than_one   =   1.0 / (1.0 - X**2) * ( -1.0 + 2.0/np.sqrt(1.0-X**2) * np.arctanh( np.sqrt((1.0 - X)/(1.0 + X)) ) )
            case_larger_than_one    =   1.0 / (X**2 - 1.0) * (  1.0 - 2.0/np.sqrt(X**2-1.0) * np.arctan(  np.sqrt((X - 1.0)/(1.0 + X)) ) )
            # return_me
            return_me               =   np.ones(np.shape(X)) * np.nan
            return_me[(X < 1.0)]    =   case_smaller_than_one[(X < 1.0) ]
            return_me[(X > 1.0)]    =   case_larger_than_one[ (X > 1.0) ]
            return_me[(X== 1.0)]    =   1.0 / 3.0
            return return_me
        # calc sigma while surpressing warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # derive sigma
            sigma   =   self.sigma_s * f2d(X = xi / xi_s)
        # return
        return sigma
    # ---
    # prokect_map
    # ---
    def Sigma_XY(self, XX, YY):
        """
        Follows eq~29 in Umetsu+15.

        However, eq~29 in Umetsu+15 describes the ellipse which is after the first two Euler angles (phi and theta).
        One needs additional third Euler angle to describe the rotation on the observer's sky.

        Denote the coordinate system after the rotation of (phi and theta) by X' and Y'. On the other hand, we denote the
        coordinate system of the observer's sky after the third rotation angle (psi) applied. The (X'', Y'') are the axes after
        rotating the axes (X', Y') by the psi angle. One will get:

        | X' |   | cos(psi) -sin(psi) |   | X'' |
        |    | = |                    | * |     |
        | Y' |   | sin(psi)  cos(psi) |   | Y'' |

        Last, replacing (X', Y') by the observer's coordinate, (X'', Y''), eq~29 in Umetsu+15 becomes

        zeta**2 = 1 / f * (j2prime * X''**2 + 2 * k2prime * X''Y'' + l2prime * Y''**2),

        where

        j2prime = ( j * cos(psi)**2 + 2 * k * cos(psi) * sin(psi) + l * sin(psi)**2 )
        k2prime = ( cos(psi) * sin(psi) * (l - j) + k * (cos(psi)**2 - sin(psi)**2) )
        l2prime = ( j * sin(psi)**2 - 2 * k * cos(psi) * sin(psi) + l * cos(psi)**2 )

        """
        # sanitize
        XX          =       np.array(XX, ndmin=1)
        YY          =       np.array(YY, ndmin=1)
        # derive zeta
        #zeta        =       np.sqrt( (self.jj * XX**2 + 2 * self.kk * XX * YY + self.ll * YY**2) / self.ff )
        zeta        =       np.sqrt( (self.j2prime * XX**2 + 2 * self.k2prime * XX * YY + self.l2prime * YY**2) / self.ff )
        # return    
        return self.Sigma(zeta = zeta)


if   __name__ == "__main__":
    A = pNFW()
    # test
    import scipy.stats as stats

    # nclusters
    nclusters   =   1000
    qa          =   np.random.uniform(0.001, 0.49, nclusters)
    #qb          =   np.ones(nclusters)
    qb          =   np.random.uniform(0.50, 1.0, nclusters)
    theta       =   np.random.uniform(-0.5*pi, 0.5*pi, nclusters)
    phi         =   np.random.uniform(0, 2*pi, nclusters)

    halo_list   =   [ pNFW(qa = qa[nn], qb = qb[nn], theta = theta[nn], phi = phi[nn]) for nn in xrange(nclusters) ]
    halo_q_proj = np.array([halo.q_proj for halo in halo_list])
    q_edges     =   np.linspace(0.0, 1.0, 100)
    q_bins      =   0.5 * (q_edges[1:] + q_edges[:-1])
    hist        =   np.histogram(halo_q_proj, bins = q_edges)[0]
    import matplotlib.pyplot as pyplt
    pyplt.plot(q_bins, hist, "k-")
    pyplt.show()

    XX, YY = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
    pyplt.imshow(A.Sigma_XY(XX, YY))
    pyplt.show()

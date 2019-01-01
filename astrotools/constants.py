import astropy.constants as const

# geometrical constants
pi = 3.141592654

# physical constants
c = const.c.value  # speed of light
G = const.G.value  # gravitational constant
h = const.h.value  # Planck constant
k_B = const.k_B.value  # Boltzmann constants

# natural constants
kpc = const.kpc.value
mpc =  kpc*1000 #Megaparsec in m

year_in_sec = 3.155815e7 #	year in seconds

Msun = const.M_sun.value

# physical constants in cgs
G_cgs = const.G.cgs.value

# natural constants in cgs
kpc_cgs = const.kpc.cgs.value
mpc_cgs = kpc*1000

Msun_cgs = const.M_sun.cgs.value

G_cosmo  = G*Msun/(1.0e6*kpc) # G in (km/s)^2 kpc msun^-1

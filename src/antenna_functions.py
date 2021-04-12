'''
A simple AntArray class which has methods to compute geometric phase delays
of source signals.
'''

from numpy import array, zeros, sin, cos, deg2rad
import datetime
import ephem

class AntArray(object):
    def __init__(self, array_loc, ant_locs):
        '''
        Initialize an array instance.
        array_loc should be a [lat, long] pair, in degrees
        ant_locs should be an antennas x 3 (xyz) list/array of antennas
        '''
        self.lat_d, self.lon_d = array_loc
        self.lat_r, self.lon_r = deg2rad(array_loc)
        self._cos_lat = cos(self.lat_r)
        self._sin_lat = sin(self.lat_r)

        self.ant_locs = array(ant_locs)
        self.n_ants = len(self.ant_locs)
        self.bls_in_m = self._get_bls_in_m()
        self.obs = self._get_ephem_obs()

    def _get_ephem_obs(self):
        '''
        Return a pyephem observer at the array's
        latitude / longitude.
        '''
        out = ephem.Observer()
        out.lat = self.lat_r
        out.lon = self.lon_r
        return out

    def _get_bls_in_m(self):
        '''
        returns an n_ants x n_ants x 3 array
        such that out[i,j] = the xyz vector
        from antenna i to j in m
        '''
        out = zeros([self.n_ants, self.n_ants, 3])
        for i in xrange(self.n_ants):
            for j in xrange(self.n_ants):
                out[i,j] = self.ant_locs[j] - self.ant_locs[i]
        return out

    def get_sidereal_time(self, t):
        self.obs.date = datetime.datetime.fromtimestamp(t)
        return self.obs.sidereal_time()

        
    def get_xyz_uvw_rot_matrix(self, ra=None, t=None, dec=None, ha=None):
        '''
        returns a rotation matrix M,
        such that [uvw vector] = M[xyz vector].
        ha and dec should be the hour angle
        and phase centre of the UV plane in
        radians.
        '''
        if (ha is None) and ((ra is None) or (t is None)):
            raise RuntimeError("Can't compute xyz -> uvw matrix without either an ha input, or both an ra and time")
        else:
            if ha is None:
                # calculate the hour angle based on ra and time
                self.obs.date = datetime.datetime.fromtimestamp(t)
                ha = self.obs.sidereal_time() - ra

            cosH = cos(ha)
            sinH = sin(ha)
            cosD = cos(dec)
            sinD = sin(dec)
            cosL = self._cos_lat
            sinL = self._sin_lat

            M = array(zeros([3,3]))
            M[0] = [cosH, -sinH*sinL, sinH*sinD]
            M[1] = [sinH*sinD, cosH*sinD*sinL + cosD*cosL, -cosH*sinD*cosL + cosD*sinL]
            M[2] = [-sinH*cosD, -cosH*cosD*sinL + sinD*cosL, cosH*cosD*cosL + sinD*sinL]

            return M

    def get_uvw_in_m(self, ra=None, t=None, dec=None, ha=None):
        '''
        returns an n_ants x n_ants x 3 array
        such that out[i,j] = the uvw vector
        from ant i to j in metres.
        '''
        M = self.get_xyz_uvw_rot_matrix(ra=ra, t=t, dec=dec, ha=ha)
        out = zeros([self.n_ants, self.n_ants, 3])
        for i in xrange(self.n_ants):
            for j in xrange(self.n_ants):
                out[i,j] = dot(M, self.bls_in_m)
        return out


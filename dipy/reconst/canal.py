import numpy as np
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_model
from dipy.reconst.shm import real_sph_harm
from scipy.special import genlaguerre, gamma, hyp2f1
from dipy.core.geometry import cart2sphere
from math import factorial
# Next step: Tester cette class


@multi_voxel_model
class AnalyticalModel(Cache):

    def __init__(self,
                 gtab):
        r""" Analytical and continuous modeling of the diffusion signal


        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab

    def fit(self, data):
        return AnalyticalFit(self, data)


class AnalyticalFit():

    def __init__(self, model, data):
        """ Calculates PDF and ODF and other properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        data : 1d ndarray,
            signal values
        """

        self.model = model
        self.data = data

        # def setReconstructionMatrix(self, radialOrder, zeta):
        #     pass

    def l2estimation(self, radialOrder, zeta):
        pass

    def pdf(self):
        """ Applies the 3D FFT in the q-space grid to generate
        the diffusion propagator
        """
        Pr = np.zeros(
            self.data.shape)  # Add another set of measurement location in the R^3 where to compute the pdf (rtab for example)

        return Pr

    def odf(self, sphere):
        r""" Calculates the real discrete odf for a given discrete sphere

        ..math::
            :nowrap:
                \begin{equation}
                    \psi_{DSI}(\hat{\mathbf{u}})=\int_{0}^{\infty}P(r\hat{\mathbf{u}})r^{2}dr
                \end{equation}

        where $\hat{\mathbf{u}}$ is the unit vector which corresponds to a
        sphere point.
        """

        Psi = np.zeros(sphere.shape[0])

        # calculate the orientation distribution function
        return Psi


@multi_voxel_model
class ShoreModel(AnalyticalModel):

    def __init__(self,
                 gtab):
        r""" Analytical and continuous modeling of the diffusion signal


        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab

    def fit(self, data):
        return ShoreFit(self, data)


class ShoreFit(AnalyticalFit):

    def __init__(self, model, data):
        """ Calculates PDF and ODF and other properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        data : 1d ndarray,
            signal values
        """

        self.model = model
        self.data = data
        self.gtab = model.gtab
        self.Sshore = None
        self.Cshore = None



    # def setReconstructionMatrix(self,radialOrder=6, zeta=700):
    #     print('ser rec')
    #     self.radialOrder = radialOrder
    #     self.zeta = zeta
    #     M = SHOREmatrix(self.radialOrder,  self.zeta, self.gtab)
    # cette matrix a besoin d'etre calculer qu'une fois (regarder l'implementation de la classe QBI)
    # print(M.shape)

    #     return M

    def l2estimation(self,radialOrder=6, zeta=700,lambdaN=1e-8,lambdaL=1e-8):

        self.radialOrder = radialOrder
        self.zeta = zeta
        Lshore = L_SHORE(self.radialOrder)
        Nshore = N_SHORE(self.radialOrder)
        

        M= self.model.cache_get('shore_matrix', key=self.gtab)
        if M is None:
			M = SHOREmatrix(self.radialOrder,  self.zeta, self.gtab)
			self.model.cache_set('shore_matrix', self.gtab, M)


        pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + lambdaN*Nshore + lambdaL*Lshore), M.T)

        # Data coefficients in SHORE basis
        self.Cshore = np.dot(pseudoInv, self.data)

        # Estimated data using the SHORE basis
        # self.Sshore = np.dot(self.Cshore, M.T)

        return self.Cshore#, self.Sshore

    def pdf(self):
        """ Applies the 3D FFT in the q-space grid to generate
        the diffusion propagator
        """
        Pr = np.zeros(
            self.data.shape)  # Add another set of measurement location in the R^3 where to compute the pdf (rtab for example)

        return Pr

    def odf(self):
        r""" Calculates the real discrete odf for a given discrete sphere

        ..math::
            :nowrap:
                \begin{equation}
                    \psi_{DSI}(\hat{\mathbf{u}})=\int_{0}^{\infty}P(r\hat{\mathbf{u}})r^{2}dr
                \end{equation}

        where $\hat{\mathbf{u}}$ is the unit vector which corresponds to a
        sphere point.
        """
        J = (self.radialOrder + 1) * (self.radialOrder + 2) / 2

        Csh = np.zeros(J)
        counter = 0;

        for n in range(self.radialOrder+1):
            for l in range(0, n+1, 2):
                for m in range(-l, l+1):

                    j = int(l + m + (2 * np.array(range(0, l, 2)) + 1).sum())

                    Cnl = ((-1)**(n - l/2))/(2.0*(4.0 * np.pi**2 * self.zeta)**(3.0/2.0)) * ((2.0 * (
                        4.0 * np.pi**2 * self.zeta)**(3.0/2.0) * factorial(n - l)) / (gamma(n + 3.0/2.0)))**(1.0/2.0)
                    Gnl = (gamma(l/2 + 3.0/2.0) * gamma(3.0/2.0 + n)) / (gamma(
                        l + 3.0/2.0) * factorial(n - l)) * (1.0/2.0)**(-l/2 - 3.0/2.0)
                    Fnl = hyp2f1(-n + l, l/2 + 3.0/2.0, l + 3.0/2.0, 2.0)
                    # float(mpmath.hyp2f1(float(a), float(b), float(c), float(z)))

                    # print "(n,l,m) = (%d,%d,%d)  %f" % (n,l,m,Fnl)
                    Csh[j] += self.Cshore[counter]*Cnl*Gnl*Fnl
                    counter += 1

        # calculate the orientation distribution function
        return Csh


def SHOREmatrix(radialOrder, zeta, gtab):
    "Compute the SHORE matrix"

    qvals = np.sqrt(gtab.bvals)
    bvecs = gtab.bvecs

    qgradients = qvals[:, None] * bvecs
    r, theta, phi = cart2sphere(qgradients[:, 0], qgradients[:, 1], qgradients[:, 2])

    M = np.zeros((r.shape[0], (radialOrder+1)*((radialOrder+1)/2)*(2*radialOrder+1)))

    # calculer le r theta phi et compare avec mon code python pour savoir si c'est correct
    counter = 0
    for n in range(radialOrder+1):
        for l in range(0, n+1, 2):
            for m in range(-l, l+1):
                # print(counter)
                # print "(n,l,m) = (%d,%d,%d)" % (n,l,m)
                # print(counter)
                M[:, counter] = \
                    real_sph_harm(m, n, theta, phi) * \
                    genlaguerre(n - l, l + 0.5)(r ** 2 / zeta) * \
                    np.exp(- r ** 2 / (2 * zeta)) * \
                    kappa(zeta, n, l) * \
                    (r ** 2 / zeta)**(l/2)

                # print(sum(genlaguerre(n - l/2,l + 0.5)(r ** 2 / zeta)))
                counter += 1
    return M[:, 0:counter]


def kappa(zeta, n, l):
    if n-l < 0:
        return np.sqrt((2 * 1) / (zeta ** 1.5 * gamma(n + 1.5)))
    else:
        return np.sqrt((2 * factorial(n - l)) / (zeta ** 1.5 * gamma(n + 1.5)))


def L_SHORE(radialOrder):
    "Returns the angular regularisation matrix for SHORE basis"
    diagL = np.zeros((radialOrder+1)*((radialOrder+1)/2)*(2*radialOrder+1))
    counter = 0;
    for n in range(radialOrder+1):
        for l in range(0, n+1, 2):
            for m in range(-l, l+1):
                # print(counter)
                # print "(n,l,m) = (%d,%d,%d)" % (n,l,m)
                # print(counter)
                diagL[counter] = (l * (l + 1)) ** 2
                counter += 1

    return np.diag(diagL[0:counter])


def N_SHORE(radialOrder):
    "Returns the angular regularisation matrix for SHORE basis"
    diagN = np.zeros((radialOrder+1)*((radialOrder+1)/2)*(2*radialOrder+1))
    counter = 0;
    for n in range(radialOrder+1):
        for l in range(0, n+1, 2):
            for m in range(-l, l+1):
                # print(counter)
                # print "(n,l,m) = (%d,%d,%d)" % (n,l,m)
                # print(counter)
                diagN[counter] = (n * (n + 1)) ** 2
                counter += 1

    return np.diag(diagN[0:counter])

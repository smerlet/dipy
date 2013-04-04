import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_model
from dipy.reconst.shm import real_sph_harm
from scipy.special import genlaguerre, gamma
from dipy.core.geometry import cart2sphere
import math
# Next step: Tester cette class


@multi_voxel_model
class AnalyticalModel():

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

    def l2estimation(self, radialOrder, zeta):
        M = SHOREmatrix(radialOrder, zeta, self.gtab)
        print(M,M.shape)

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
        return np.sqrt((2 * math.factorial(n - l)) / (zeta ** 1.5 * gamma(n + 1.5)))

import numpy as np
from dipy.data import get_data
from dipy.reconst.canal import ShoreModel
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.data import get_sphere
from dipy.sims.voxel import SticksAndBall


def test_canal():

    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')

    # Load gradient table
    fbvals, fbvecs = get_data('2shellstable')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs) #Il faut traiter le cas ou le signal n'est pas normalise
    gtab = gradient_table(bvals[1:], bvecs[1:,:])

    # Generate synthetic data
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)

    # Create an ShoreModel object
    asm = ShoreModel(gtab)

    # Fit the analytical model to the data
    asmfit = asm.fit(data)


    # Estimate the SHORE coefficient using a l2 regularization
    radialOrder=4
    zeta=700
    asmfit.l2estimation(radialOrder, zeta)

    # Evaluate the ODF on the direction provided by 'sphere'
    odf = asmfit.odf(sphere.vertices)

    print(odf.shape)

if __name__ == '__main__':
    # run_module_suite()
    test_canal()

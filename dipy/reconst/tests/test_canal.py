import numpy as np
import pylab as py
from dipy.data import get_data, two_shells_voxels, get_sphere
from dipy.reconst.canal import ShoreModel
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.sims.voxel import SticksAndBall, multi_tensor
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
import nibabel as nib

def test_canal():

    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')

    # Load gradient table
    fimg, fbvals, fbvecs = get_data('ISBI_testing_2shells_table')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)  # Il faut traiter le cas ou le signal n'est pas normalise
    gtab = gradient_table(bvals[1:], bvecs[1:,:])

    # Generate synthetic data
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=1, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)

    # Eigen values of the multitensor model
    mevals = np.array([[300, 1700, 300], [300, 1700, 300]])*1e-6
    data, golden_directions = multi_tensor(gtab, np.array(mevals), S0=1, angles=[(0, 0), (70, 0)],
                     fractions=[60, 40], snr=20)

    # Create an ShoreModel object
    asm = ShoreModel(gtab)

    # Fit the analytical model to the data
    asmfit = asm.fit(data)

    # Estimate the SHORE coefficient using a l2 regularization
    radialOrder = 4
    zeta = 700
    Cshore, Sshore = asmfit.l2estimation(radialOrder=radialOrder, zeta=zeta, lambdaN=1e-8, lambdaL=1e-8)

    # py.figure()
    # py.plot(Sshore, 'r')
    # py.plot(data, 'b')
    # py.show()

    # Compute the spherical harmonic coefficients representing the ODF
    Csh = asmfit.odf()

    # Set the sh_order to compute the ODF
    sh_order = radialOrder

    # Evaluate the ODF in the direction provided by 'sphere'
    odf = sh_to_sf(Csh, sphere, sh_order, basis_type="fibernav")

    # verifier l'odf maintenant

    r = fvtk.ren()
    fvtk.add(r, fvtk.sphere_funcs(odf,
                                  sphere, colormap='jet'))

    fvtk.show(r)


def test_multivox_canal():

    data, affine, gtab = two_shells_voxels(20, 25, 35, 40, 25, 28)
    # Create an ShoreModel object
    asm = ShoreModel(gtab)

    # Fit the analytical model to the data
    asmfit = asm.fit(data)

    # Estimate the SHORE coefficient using a l2 regularization
    radialOrder = 4
    zeta = 700
    # asmfit.setReconstructionMatrix(radialOrder=radialOrder, zeta=zeta)
    Cshore = asmfit.l2estimation(radialOrder=radialOrder, zeta=zeta, lambdaN=1e-8, lambdaL=1e-8)
    print Cshore.shape
    # print res[0, 0, 0][0].shape, res[0, 0, 0][1].shape

    # Cshore=res[:,:,:][0] ca marche pas ce truc, faut trouver le moyen d'extraire chaque premier tableau, sinon enlever Sshore qui n'est pas super important
    # Sshore=res[:,:,:][1]

    # print Cshore.shape, Sshore.shape

    # for i in range(5):
                # for j in range(5):

                        # py.figure()
                        # py.plot(res[i,j,0][1], 'r')
                        # py.plot(data[i,j,0], 'b')
                        # py.show()




    # Save the result as a new Nifti file
    img = nib.Nifti1Image(Cshore.astype('f4'), affine)
    nib.save(img, 'Cshore.nii.gz')

    # Save a GFA map (For the moment, it is a black map)
    img = nib.Nifti1Image(np.ones((5,5,3)).astype('f4'), affine)
    nib.save(img, 'GFAshore.nii.gz')

    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')

    Csh = asmfit.odf()
    print(Csh.shape)
	
    # Save the result as a new Nifti file
    img = nib.Nifti1Image(Csh.astype('f4'), affine)
    nib.save(img, 'Csh.nii.gz')

    # Set the sh_order to compute the ODF
    sh_order = radialOrder

    # Evaluate the ODF in the direction provided by 'sphere'
    odf = sh_to_sf(Csh, sphere, sh_order, basis_type="fibernav")

    # verifier l'odf maintenant

    r = fvtk.ren()
    fvtk.add(r, fvtk.sphere_funcs(odf, sphere, colormap='jet'))

    fvtk.show(r)


if __name__ == '__main__':
    # run_module_suite()
    test_multivox_canal()

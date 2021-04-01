import os
import shutil

from .utils import run_bash
from .path import get_path, remove_ext, get_filename, make_dirs

import nibabel as nib

# def compute_halfway_registration(A_fp, B_fp, ndims=3, remove_tmp=True):
#     base_path = get_path(A_fp)
#
#     prefix = os.path.join(base_path, 'half')
#     reg = 'antsRegistration'
#     aff = ' -t affine[ 0.25 ]  -c [ 1009x200x20,1.e-8,20 ]  -s 4x2x0 -f 4x2x1 '
#
#     nmA = f'{prefix}_A_norm'
#     nmB = f'{prefix}_B_norm'
#     initA = f'{prefix}_initA'
#     initB = f'{prefix}_initB'
#
#     # register in both directions, then average the result
#     run_bash(f'{reg} -d {ndims} -r [ {A_fp}, {B_fp}, 1 ] -m mattes[ {A_fp}, {B_fp}, 1 , 32 , regular , 0.25 ] ' +
#              f'{aff} -z 1 -o [ {initA} ]')
#     run_bash(f'{reg} -d {ndims} -r [ {B_fp}, {A_fp}, 1 ] -m mattes[ {B_fp}, {A_fp}, 1 , 32 , regular , 0.25 ] ' +
#              f'{aff} -z 1 -o [ {initB} ]')
#     # get the identity map
#     run_bash(f'ComposeMultiTransform {ndims} {initA}_id.mat -R {initA}0GenericAffine.mat ' +
#              f' {initA}0GenericAffine.mat -i {initA}0GenericAffine.mat')
#     # invert the 2nd affine registration map
#     run_bash(f'ComposeMultiTransform {ndims} {initB}_inv.mat -R {initA}0GenericAffine.mat -i {initB}0GenericAffine.mat')
#     # get the average affine map
#     run_bash(f'AverageAffineTransform {ndims} {prefix}_avg.mat  {initB}_inv.mat {initA}0GenericAffine.mat')
#     # get the midpoint affine map
#     run_bash(f'AverageAffineTransform {ndims} {prefix}_mid.mat   {initA}_id.mat  {prefix}_avg.mat')
#
#     # .........#
#     # this applies, to B_fp, A_fp map from B_fp to midpoint(B_fp,A_fp)
#     run_bash(f'antsApplyTransforms -d {ndims} -i {B_fp} -o {prefix}_mid.nii.gz -t  {prefix}_mid.mat  -r  {A_fp}')
#     # compute the map from A_fp to midpoint(B_fp,A_fp)
#     run_bash(f'{reg} -d {ndims} -r  [ {prefix}_mid.nii.gz, {A_fp}, 1 ] -m mattes[  {prefix}_mid.nii.gz, ' +
#              f'{A_fp}, 1 , 32, random , 0.25 ] {aff} -n BSpline[ 3 ] -o [ {nmA}, {nmA}_aff.nii.gz]')
#     # compute the map from B_fp to midpoint(B_fp,A_fp) --- "fair" interpolation
#     run_bash(f'{reg} -d {ndims}  -r [ {nmA}_aff.nii.gz, {B_fp}, 1 ] -m mattes[  {nmA}_aff.nii.gz, ' +
#              f'{B_fp}, 1 , 32, random , 0.25 ] {aff} -n BSpline[ 3 ] -o [ {nmB},{nmB}_aff.nii.gz]')
#
#     if remove_tmp:
#         os.remove(os.path.join(base_path, f'{prefix}_avg.mat'))
#         os.remove(os.path.join(base_path, f'{prefix}_mid.nii.gz'))
#         os.remove(os.path.join(base_path, f'{prefix}_mid.mat'))
#         os.remove(f'{initA}_id.mat')
#         os.remove(f'{initB}_inv.mat')
#         os.remove(f'{initA}0GenericAffine.mat')
#         os.remove(f'{initB}0GenericAffine.mat')
#
#     mida_nifti = os.path.join(base_path, f'{nmA}_aff.nii.gz')
#     midb_nifti = os.path.join(base_path, f'{nmB}_aff.nii.gz')
#     mida_aff = os.path.join(base_path, f'{nmA}0GenericAffine.mat')
#     midb_aff = os.path.join(base_path, f'{nmB}0GenericAffine.mat')
#     return mida_nifti, midb_nifti, mida_aff, midb_aff


def perform_halfway_registration(
        baseline_fpath,
        followup_fpath,
        baseline_half_fpath,
        followup_half_fpath,
        baseline2half_fpath,
        followup2half_fpath):

    # Get dimensions names and paths
    ndims = nib.load(baseline_fpath).header['dim'][0]
    bname = get_filename(baseline_fpath, ext=False)
    fname = get_filename(followup_fpath, ext=False)

    tmp_path = get_path(baseline_fpath)

    # Declare names for intermediate stages
    B, F = baseline_fpath, followup_fpath
    Bhalf = os.path.join(tmp_path, f'{bname}_half')
    Fhalf = os.path.join(tmp_path, f'{fname}_half')
    Binit = os.path.join(tmp_path, f'{bname}_init_half')
    Finit = os.path.join(tmp_path, f'{fname}_init_half')

    # Generate the filepaths that will be output from ANTS functions
    Binit_id_fp = f'{Binit}_id.mat'
    Finit_inv_fp = f'{Finit}_inv.mat'
    Binit2half_fp = f'{Binit}0GenericAffine.mat'
    Finit2half_fp = f'{Finit}0GenericAffine.mat'

    half_avg_tx_fp = os.path.join(tmp_path, f'half_avg.mat')
    half_avg_fp = os.path.join(tmp_path, f'half.nii.gz')
    half_mid_fp = os.path.join(tmp_path, f'half_mid.mat')

    Bhalf_fp = f'{Bhalf}_aff.nii.gz'
    Fhalf_fp = f'{Fhalf}_aff.nii.gz'
    B2half_fp = f'{Bhalf}0GenericAffine.mat'
    F2half_fp = f'{Fhalf}0GenericAffine.mat'

    # Predeclare useful comand parts
    reg = f'antsRegistration -d {ndims}'
    aff = ' -t affine[ 0.25 ]  -c [ 1009x200x20,1.e-8,20 ]  -s 4x2x0 -f 4x2x1 '

    ### Register in both directions, then average the result
    run_bash(
        f'{reg} -r [ {B}, {F}, 1 ] -m mattes[ {B}, {F}, 1 , 32 , regular , 0.25 ] {aff} -z 1 -o [ {Binit} ]', v=False)
    run_bash(
        f'{reg} -r [ {F}, {B}, 1 ] -m mattes[ {F}, {B}, 1 , 32 , regular , 0.25 ] {aff} -z 1 -o [ {Finit} ]', v=False)
    # get the identity map
    run_bash(
        f'ComposeMultiTransform {ndims} {Binit_id_fp} -R {Binit2half_fp}  {Binit2half_fp} -i {Binit2half_fp}', v=False)
    # invert the 2nd affine registration map
    run_bash(f'ComposeMultiTransform {ndims} {Finit_inv_fp} -R {Binit2half_fp} -i {Finit2half_fp}', v=False)
    # get the average affine map
    run_bash(f'AverageAffineTransform {ndims} {half_avg_tx_fp}  {Finit_inv_fp} {Binit2half_fp}', v=False)
    # get the midpoint affine map
    run_bash(f'AverageAffineTransform {ndims} {half_mid_fp} {Binit_id_fp}  {half_avg_tx_fp}', v=False)

    # this applies, to F, B map from F to midpoint(F,B)
    run_bash(f'antsApplyTransforms -d {ndims} -i {F} -o {half_avg_fp} -t {half_mid_fp} -r {B}', v=False)
    # compute the map from B to midpoint(F,B)
    run_bash(f'{reg} -r  [ {half_avg_fp}, {B}, 1 ] -m mattes[ {half_avg_fp}, {B}, 1 , 32, random , 0.25 ] ' +
             f'{aff} -n BSpline[ 3 ] -o [ {Bhalf}, {Bhalf_fp}]', v=False)
    # compute the map from F to midpoint(F,B)
    run_bash(f'{reg} -r [ {Bhalf_fp}, {F}, 1 ] -m mattes[ {Bhalf_fp}, {F}, 1 , 32, random , 0.25 ] ' +
             f'{aff} -n BSpline[ 3 ] -o [ {Fhalf}, {Fhalf_fp}]', v=False)

    # Ensure output directories exist before copying files
    for fpath in [baseline_half_fpath, followup_half_fpath, baseline2half_fpath, followup2half_fpath]:
        make_dirs(get_path(fpath))

    shutil.copy(B2half_fp, baseline2half_fpath)
    shutil.copy(F2half_fp, followup2half_fpath)

    # TODO load nibabel instead of copy for images and transform to float
    shutil.copy(Bhalf_fp, baseline_half_fpath)
    shutil.copy(Fhalf_fp, followup_half_fpath)

    # Remove all temporary files
    to_remove = [half_avg_tx_fp, half_avg_fp, half_mid_fp, Binit_id_fp, Finit_inv_fp, Binit2half_fp, Finit2half_fp,
                 Bhalf_fp, Fhalf_fp, B2half_fp, F2half_fp]
    [os.remove(fp) for fp in to_remove]


def apply_linear_transform(in_fp, ref_fp, tx_fp, out_fp, interp='Linear', out_value=0, out_dtype='float', ndims=3):
    apply_tx_cmd = f'antsApplyTransforms -d {ndims} -i {interp} -f {out_value} -u {out_dtype}'
    run_bash(f'{apply_tx_cmd} -i {in_fp} -o {out_fp} -t {tx_fp} -r {ref_fp}', v=False)


def perform_nonlinear_registration(
        baseline_fpath,
        followup_fpath,
        baseline_initial_tx,
        followup_initial_tx,
        image_out_fpath,
        fields_out_fpath=None,
        fields_inv_out_fpath=None,
        erase_image=False,
        erase_ants_fields=True,
        ndims=3):
    reg = f'antsRegistration -d {ndims}'
    initial_transforms = \
        f'--initial-fixed-transform {baseline_initial_tx} --initial-moving-transform {followup_initial_tx}'
    metric = f'-m mattes[ {baseline_fpath}, {followup_fpath} , 1 , 32 ]'
    transform = f'-t syn[ 0.25, 3, 0.0 ] -c [ 20x20x10,1.e-8,10 ] -s 2x1x0 -f 4x2x1'
    output = f'-o [{remove_ext(image_out_fpath)}, {image_out_fpath}]'
    run_bash(cmd=f'{reg} {initial_transforms} {metric} {transform} {output}', v=False)

    fields_ants_fpath = f'{remove_ext(image_out_fpath)}1Warp.nii.gz'
    if fields_out_fpath is not None:
        shutil.copy(fields_ants_fpath, fields_out_fpath)
        os.remove(fields_ants_fpath)
    elif erase_ants_fields:
        os.remove(fields_ants_fpath)

    fields_inv_ants_fpath = f'{remove_ext(image_out_fpath)}1InverseWarp.nii.gz'
    if fields_inv_out_fpath is not None:
        shutil.copy(fields_inv_ants_fpath, fields_inv_out_fpath)
        os.remove(fields_inv_ants_fpath)
    elif erase_ants_fields:
        os.remove(fields_inv_ants_fpath)

    if erase_image:
        os.remove(image_out_fpath)


def compute_jacobian(fields_fp, jacobian_out_fp, ndims=3):
    do_logjacobian = 0
    use_geometric = 1
    run_bash(f'CreateJacobianDeterminantImage {ndims} {fields_fp} {jacobian_out_fp} {do_logjacobian} {use_geometric}',
             v=False)
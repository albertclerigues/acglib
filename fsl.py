import subprocess
import os
import shutil
import nibabel as nib

from .path import remove_ext, make_dirs, get_path, get_filename

def run_fast(filepath_in):
    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    pve_fpaths = [remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_seg.nii.gz'))



def run_first(filepath_in, is_skull_stripped=True):
    print('Running FIRST: {}'.format(filepath_in))
    first_fpath_out = remove_ext(filepath_in) + '_first.nii.gz'

    # Create temporary path
    tmp_path = make_dirs(os.path.join(get_path(filepath_in), 'first_tmp'))
    t1_fpath = os.path.join(tmp_path, get_filename(filepath_in, ext=True))
    first_fpath_in = os.path.join(tmp_path, 't1_all_fast_firstseg.nii.gz')

    shutil.copy(filepath_in, t1_fpath)
    if is_skull_stripped:
        first_cmd_template = 'run_first_all -b -i {} -o t1'
    else:
        first_cmd_template = 'run_first_all -i {} -o t1'

    subprocess.check_call(['bash', '-c', first_cmd_template.format(t1_fpath)], cwd=tmp_path)

    first_nifti = nib.load(first_fpath_in)
    nib.Nifti1Image(
        (first_nifti.get_fdata() > 0.5).astype(float), first_nifti.affine, first_nifti.header
    ).to_filename(first_fpath_out)

    # Erase all temporary files
    shutil.rmtree(tmp_path)
    
def register_to_mni(t1_filepath, reference_filepath, transform_filepath_out, reg_filepath_out=None):
    """Registers the image to MNI space and stores the transform to MNI"""
    register_cmd = 'flirt -in {} -ref {} -omat {} '.format(t1_filepath, reference_filepath, transform_filepath_out)
    if reg_filepath_out is not None:
        register_cmd += '-out {} '.format(reg_filepath_out)
    register_opts = '-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'
    subprocess.check_output(['bash', '-c', register_cmd + register_opts])


def segment_tissue(filepath_in):
    """Performs 3 tissue segmentation"""

    print('Running FAST: {}'.format(filepath_in))
    subprocess.check_call(['bash', '-c', 'fast {}'.format(filepath_in)])

    pve_fpaths = [remove_ext(filepath_in) + '_pve_{}.nii.gz'.format(i) for i in range(3)]
    out_fpaths = [remove_ext(filepath_in) + '_fast_{}.nii.gz'.format(i) for i in range(3)]

    for pve_fpath, out_fpath in zip(pve_fpaths, out_fpaths):
        os.rename(pve_fpath, out_fpath)

    # Remove all other files
    os.remove(os.path.join(remove_ext(filepath_in) + '_mixeltype.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_pveseg.nii.gz'))
    os.remove(os.path.join(remove_ext(filepath_in) + '_seg.nii.gz'))
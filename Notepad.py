import os
import shutil
import nibabel as nib
import glob
import cv2
import numpy as np

path_base_input = r'C:\Users\hamid\D_drive\University\Ryan\Data'
folders = glob.glob(path_base_input + '/*')
files = ['B1map_T1space', 'B1map_T1space_brain', 'T1_biascorr_brain', 'T1_biascorr_brain_mask2',
        'T1_fast_pve_0', 'T1_fast_pve_1_0p9', 'T1_fast_pve_2_0p9']


for fo in folders:
    for fi in files:
        path_file = os.path.join(fo, fi + '.nii.gz')

        data_3d = nib.load(path_file).get_fdata()

        if not os.path.isdir(fo + '/axial'):
            os.mkdir(fo + '/axial')

        for i in range(data_3d.shape[2]):
            img = data_3d[:,:,i]
            name = fi + '_{}.png'.format(i)

            img = ( (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10) * (2**16 - 1) ).astype('uint16')

            cv2.imwrite(fo + '/axial/' + name, img)
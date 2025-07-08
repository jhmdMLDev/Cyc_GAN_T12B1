import os
from os import listdir
from os.path import isfile, join
import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import nibabel as nib
from PIL import Image
# from scipy import ndimage


def pad_array(T1, B1, pad):
    pad_np = ((pad[0]//2, int(np.ceil(pad[0]/2))), (pad[1]//2, int(np.ceil(pad[1]/2))), )
    T1 = np.pad(T1, pad_np)
    B1 = np.pad(B1, pad_np)
    return T1, B1


def crop_3D(input, mask, image_size):
    cropped_inputs_T1 = []
    cropped_masks = []
    
    # min0, min1, min2 = (np.ceil(image_size[0]/2), np.ceil(image_size[1]/2), np.ceil(image_size[2]/2))
    #N0, N1, N2 = ( np.array(input.shape) / np.array(image_size) ).astype('int')
    N0 = np.arange(0, input.shape[0], image_size[0])
    N1 = np.arange(0, input.shape[1], image_size[1])
    N2 = np.arange(0, input.shape[2], image_size[2])
    for i in range(len(N0) - 1):
        for j in range(len(N1) - 1):
            for k in range(len(N2) - 1):
                cropped_inputs_T1.append(input[N0[i]:N0[i+1], N1[j]:N1[j+1], N2[k]:N2[k+1]][np.newaxis])
                cropped_masks.append(mask[N0[i]:N0[i+1], N1[j]:N1[j+1], N2[k]:N2[k+1]][np.newaxis])

    return np.vstack(cropped_inputs_T1), np.vstack(cropped_masks)


def normalize_data(input, mask, p_h=0, p_v=0):

    transformation = transforms.Compose([
            #transforms.Resize((208,208)),
            transforms.RandomHorizontalFlip(p=p_h),
            transforms.RandomVerticalFlip(p=p_v),
            transforms.ToTensor(),
    ])

    input_transformed = transformation(Image.fromarray(input))
    mask_transformed = transformation(Image.fromarray(mask))

    return input_transformed, mask_transformed


class Pair_T1B1_Dataset(data.Dataset):
    def __init__(self, dir_dataset, image_size, n_train, pad=[0,0,0], degrade=False, perspective=False, crop=False,
                mask_roi=True, zoom=[0,0,0]):
        super(Pair_T1B1_Dataset, self).__init__()

        self.degrade = degrade
        self.perspective = perspective
        self.crop = crop
        self.image_size = image_size
        self.pad = pad
        self.mask_roi = mask_roi
        self.zoom = zoom
        self.path_base = dir_dataset

        self.N_axial = 196
        
        self.dir_T1_nifti = []
        self.folders = []
        for n in n_train:
            name = 'Ab300_{:03d}'.format(n)
            if os.path.isfile(join(join(dir_dataset,name), 'T1_biascorr_brain.nii.gz')):
                folder_imgs = join(dir_dataset,name,'axial')
                self.dir_T1_nifti += glob.glob(folder_imgs + '/B1map_T1space_brain_*.png')
                self.folders.append(n)

            

    def __getitem__(self, index):
        p_h = np.random.randint(2)
        p_v = np.random.randint(2)

        f = int(index//self.N_axial + 1)
        folder_ind = self.folders[f]
        ind = int(index - (f-1)*self.N_axial)

        folder = 'Ab300_{:03d}'.format(folder_ind)

        path_to_images = join(self.path_base, folder, 'axial')

        T1 = cv2.imread(path_to_images + '/T1_biascorr_brain_{}.png'.format(ind), 0)
        B1 = cv2.imread(path_to_images + '/B1map_T1space_brain_{}.png'.format(ind), 0)
        brain = cv2.imread(path_to_images + '/T1_biascorr_brain_mask2_{}.png'.format(ind), 0)
        if self.mask_roi:
            csf = cv2.imread(path_to_images + '/T1_fast_pve_0_{}.png'.format(ind), 0)
            wm = cv2.imread(path_to_images + '/T1_fast_pve_1_0p9_{}.png'.format(ind), 0)
            gm = cv2.imread(path_to_images + '/T1_fast_pve_2_0p9_{}.png'.format(ind), 0)

        if np.sum(self.pad)>0:
            T1, B1 = pad_array(T1, B1, self.pad)
            brain, _ = pad_array(brain, brain, self.pad)
            if self.mask_roi:
                csf, _ = pad_array(csf, csf, self.pad)
                wm, _ = pad_array(wm, wm, self.pad)
                gm, _ = pad_array(gm, gm, self.pad)


        if self.crop:
            T1, B1 = crop_3D(T1, B1, self.image_size)
            if self.mask_roi:
                csf, _ = crop_3D(csf, csf, self.image_size)
                wm, _ = crop_3D(wm, wm, self.image_size)
                gm, _ = crop_3D(gm, gm, self.image_size)


        if self.degrade:
            # T1 = degrade(T1)
            pass

        if self.perspective:
            k = np.random.choice(2,2, replace=False)
            T1 = np.transpose(T1, (k[0], k[1]))
            B1 = np.transpose(B1, (k[0], k[1]))
            if self.mask_roi:
                csf = np.transpose(csf, (k[0], k[1]))
                wm = np.transpose(wm, (k[0], k[1]))
                gm = np.transpose(gm, (k[0], k[1]))

        #brain = torch.unsqueeze(torch.from_numpy(brain), 0)

        T1, B1 = normalize_data(T1, B1, p_h, p_v)
        brain, _ = normalize_data(brain, brain, p_h, p_v)
        T1 = T1 * brain
        B1 = B1 * brain

        if self.mask_roi:
            csf, _ = normalize_data(csf, csf, p_h, p_v)
            wm, _ = normalize_data(wm, wm, p_h, p_v)
            gm, _ = normalize_data(gm, gm, p_h, p_v)

            inputs_T1 = torch.cat((T1, csf, wm, gm), 0)
            outputs_B1 = torch.cat((B1, csf, wm, gm), 0)
        else:
            inputs_T1 = T1
            outputs_B1 = B1

        return inputs_T1.float(), outputs_B1.float(), brain.float()

    def __len__(self):
        return len(self.dir_T1_nifti)



def preview_dataloader(dir_folder, pad=[0,0,0], mask_roi=True, zoom=[0,0,0], N_axial=196):
    test_set = [80,90,100,110,120,130]

    whole_test_set_T1 = []
    whole_test_set_B1 = []

    for ind in test_set:

        path_to_images = join(dir_folder, 'axial')

        T1 = cv2.imread(path_to_images + '/T1_biascorr_brain_{}.png'.format(ind), 0)
        B1 = cv2.imread(path_to_images + '/B1map_T1space_brain_{}.png'.format(ind), 0)
        brain = cv2.imread(path_to_images + '/T1_biascorr_brain_mask2_{}.png'.format(ind), 0)
        if mask_roi:
            csf = cv2.imread(path_to_images + '/T1_fast_pve_0_{}.png'.format(ind), 0)
            wm = cv2.imread(path_to_images + '/T1_fast_pve_1_0p9_{}.png'.format(ind), 0)
            gm = cv2.imread(path_to_images + '/T1_fast_pve_2_0p9_{}.png'.format(ind), 0)

        if np.sum(pad)>0:
            T1, B1 = pad_array(T1, B1, pad)
            brain, _ = pad_array(brain, brain, pad)
            if mask_roi:
                csf, _ = pad_array(csf, csf, pad)
                wm, _ = pad_array(wm, wm, pad)
                gm, _ = pad_array(gm, gm, pad)


        T1, B1 = normalize_data(T1, B1)
        brain, _ = normalize_data(brain, brain)
        T1 = T1 * brain
        B1 = B1 * brain

        if mask_roi:
            csf, _ = normalize_data(csf, csf)
            wm, _ = normalize_data(wm, wm)
            gm, _ = normalize_data(gm, gm)

            inputs_T1 = torch.cat((T1, csf, wm, gm), 0)
            outputs_B1 = torch.cat((B1, csf, wm, gm), 0)
        else:
            inputs_T1 = T1
            outputs_B1 = B1
        
        whole_test_set_T1.append(torch.unsqueeze(inputs_T1, 0))
        whole_test_set_B1.append(torch.unsqueeze(B1, 0))

    whole_test_set_T1 = torch.cat(whole_test_set_T1, 0)
    whole_test_set_B1 = torch.cat(whole_test_set_B1, 0)

    #######################
    for i in range(len(whole_test_set_T1)):
        tmp = whole_test_set_T1[0,i,:,:].detach().cpu().numpy()
        tmp = ( (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-10) * (2**16 - 1) ).astype('uint16')
        cv2.imwrite(r'C:\Users\hamid\D_drive\University\Ryan\Misc\input_{}.png'.format(i), tmp)

        tmp = whole_test_set_B1[0,i,:,:].detach().cpu().numpy()
        tmp = ( (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-10) * (2**16 - 1) ).astype('uint16')
        cv2.imwrite(r'C:\Users\hamid\D_drive\University\Ryan\Misc\output_{}.png'.format(i), tmp)

    return whole_test_set_T1.float(), whole_test_set_B1.float()

    
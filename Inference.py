import os
import torch
from Data_loader import preview_dataloader
import nibabel as nib
import numpy as np
from models import VGG_encoder, VGG_decoder, Discriminator

path_data = "/scratch/javadhe/Data/Ab300_390/"
path_model = "/scratch/javadhe/Models/Cyc_GAN/_10/"
path_test_save = "/scratch/javadhe/Result_Test/Cyc_GAN/_10/"
checkpoint = 80
pad = [0,0,12]
mask_roi = True
in_channels = 4 if mask_roi else 1
features = 1

device = 'cpu'

T1, B1, affine, mx_T1 = preview_dataloader(path_data, pad, mask_roi)

T1 = torch.unsqueeze(T1,0)
T1 = T1.to(device)

path_encoder = path_model + '/ENC_T1__10_{}.model'.format(checkpoint)
path_decoder = path_model + '/DEC_T1B1__10_{}.model'.format(checkpoint)

encoder_T1 = VGG_encoder(features=features, in_channels=in_channels).to(device)
decoder_T1B1 = VGG_decoder(features=features).to(device)
# encoder_B1 = VGG_encoder(features=features, in_channels=in_channels).to(device)
# decoder_B1T1 = VGG_decoder(features=features).to(device)
# disc_T1 = Discriminator(features=features).to(device)
# disc_B1 = Discriminator(features=features).to(device)

encoder_T1.load_state_dict(torch.load(path_encoder))
decoder_T1B1.load_state_dict(torch.load(path_decoder))

latent = encoder_T1(T1)
output = decoder_T1B1(latent)

output = np.squeeze(output.detach().cpu().numpy())

img = nib.Nifti1Image(output, np.eye(4))
nib.save(img, os.path.join(path_test_save, 'B1_pred_Inference.nii.gz'))  
img = nib.Nifti1Image(B1 - output, np.eye(4))
nib.save(img, os.path.join(path_test_save, 'B1_pred_Inference.nii.gz'))  
if not os.path.isfile(os.path.join(path_test_save, 'B1_test.nii.gz')):
    img = nib.Nifti1Image(B1, np.eye(4))
    nib.save(img, os.path.join(path_test_save, 'B1_test.nii.gz'))  
import numpy as np
import nibabel as nib
import os
from models import VGG_decoder, VGG_encoder, Discriminator
import torch
from torchsummary import summary


dir_nii = r"C:\Users\hamid\D_drive\University\Ryan\Misc\T1.nii.gz"
path_save = r"C:\Users\hamid\D_drive\University\Ryan\Misc\T1_debug_aff.nii.gz"
path_model_enc = r"C:\Users\hamid\D_drive\University\Ryan\Misc\Model_test\ENC_T1__10_80.model"
path_model_dec = r"C:\Users\hamid\D_drive\University\Ryan\Misc\Model_test\DEC_T1B1__10_80.model"
path_model_disc = r"C:\Users\hamid\D_drive\University\Ryan\Misc\Model_test\DISC_B1__10_brain_40.model"

pred_nii = nib.load(dir_nii)
pred = pred_nii.get_fdata()
B1 = nib.load(r"C:\Users\hamid\D_drive\University\Ryan\B1_pred_80.nii.gz").get_fdata()
affine = pred_nii.affine
#pred = pred[:,:,:,0]

img = nib.Nifti1Image(pred, affine)
nib.save(img, path_save)

features = 1
encoder_T1 = VGG_encoder(features=features, in_channels=4)
#encoder_T1.load_state_dict(torch.load(path_model_enc), strict=False)
disc_T1 = Discriminator(features=features)
#disc_T1.load_state_dict(torch.load(path_model_disc), strict=False)
decoder_T1B1 = VGG_decoder(features=features)
#decoder_T1B1.load_state_dict(torch.load(path_model_dec))
print('*'*8,'Encoder','*'*8)
summary(encoder_T1, torch.rand(1,4,192,192,192))
print('*'*8,'Decoder','*'*8)
summary(decoder_T1B1, torch.rand(1,16 * features,14,14,14))
print('*'*8,'Discriminator','*'*8)
summary(disc_T1, torch.rand(1,2,192,192,192))
a = torch.load(path_model_dec)


input = torch.from_numpy(np.concatenate((pred[np.newaxis], B1[np.newaxis]), 0))
input = torch.unsqueeze(input, 0).float()

p = encoder_T1(input)

print('End')
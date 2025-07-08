import torch
import numpy as np
from torch.nn import L1Loss, MSELoss
from utils import L1_loss
from torch.utils.data import DataLoader
from Data_loader import Pair_T1B1_Dataset
from models import VGG_encoder, VGG_decoder, Discriminator
from trainer import train, preview_results
from configs import *
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu_ids", help="GPU device number")
# args = parser.parse_args()

# gpu_ids = []
# if args.gpu_ids is not None:
#     for g in args.gpu_ids:
#         gpu_ids.append(int(g)) 
# print(type(gpu_ids))

gpu_ids = [0,1] #,2,3]
print(gpu_ids)

in_channels = 4 if mask_roi else 1

n_train = list(range(1,N_split))
n_test = list(range(N_split, 391))


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(available_gpus)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#device = 'cpu'

dataset_train = Pair_T1B1_Dataset(dir_dataset, image_size, n_train, pad=pad, degrade=degrade, perspective=perspective, 
                crop=crop, mask_roi=mask_roi, zoom=zoom)
dataset_test = Pair_T1B1_Dataset(dir_dataset, image_size, n_test, pad=pad, mask_roi=mask_roi, zoom=zoom)
print('Dataset')

dataloader_train = DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=batch, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=batch, shuffle=True, drop_last=True)
print('DataLoader')

encoder_T1 = VGG_encoder(features=features, in_channels=in_channels, maxpool=False, avgpool=False,).to(device)
decoder_T1B1 = VGG_decoder(features=features).to(device)
encoder_B1 = VGG_encoder(features=features, in_channels=in_channels, maxpool=False, avgpool=False,).to(device)
decoder_B1T1 = VGG_decoder(features=features).to(device)
disc_T1 = Discriminator(features=features).to(device)
disc_B1 = Discriminator(features=features).to(device)

if load_check:
    path_base = "/scratch/javadhe/Models/Cyc_GAN/_10/"
    encoder_T1.load_state_dict(torch.load(path_base + '/ENC_T1__10_{}.model'.format(checkpoint)))
    decoder_T1B1.load_state_dict(torch.load(path_base + '/DEC_T1B1__10_{}.model'.format(checkpoint)))
    encoder_B1.load_state_dict(torch.load(path_base + '/ENC_B1__10_{}.model'.format(checkpoint)))
    decoder_B1T1.load_state_dict(torch.load(path_base + '/DEC_B1T1__10_{}.model'.format(checkpoint)))
    disc_T1.load_state_dict(torch.load(path_base + '/DISC_T1__10_{}.model'.format(checkpoint)))
    disc_B1.load_state_dict(torch.load(path_base + '/DISC_B1__10_{}.model'.format(checkpoint)))

# For multi gpu training
# encoder_T1 = torch.nn.DataParallel(encoder_T1, device_ids=gpu_ids).to(device) #[0,1,2,3]
# decoder_T1B1 = torch.nn.DataParallel(decoder_T1B1, device_ids=gpu_ids).to(device)
# encoder_B1 = torch.nn.DataParallel(encoder_B1, device_ids=gpu_ids).to(device)
# decoder_B1T1 = torch.nn.DataParallel(decoder_B1T1, device_ids=gpu_ids).to(device)
# disc_T1 = torch.nn.DataParallel(disc_T1, device_ids=gpu_ids).to(device)
# disc_B1 = torch.nn.DataParallel(disc_B1, device_ids=gpu_ids).to(device)

print('*'*8, 'Encoder','*'*8)
print(encoder_T1)
print('*'*8, 'Decoder','*'*8)
print(decoder_T1B1)

optimizer = torch.optim.SGD(list(encoder_T1.parameters()) + list(decoder_T1B1.parameters()) + \
                            list(encoder_B1.parameters()) + list(decoder_B1T1.parameters()),
                            lr, momentum=0.9)
optimizer_disc = torch.optim.SGD(list(disc_T1.parameters()) + list(disc_B1.parameters()), 
                            lr, momentum=0.9)

criterion_cyc = L1_loss()
criterion_GAN = MSELoss()

torch.cuda.empty_cache()
for e in range(epoch):

    for disc in  [True, False]: 
        train(dataloaders = [dataloader_train, dataloader_test], 
            model_enc = [encoder_T1, encoder_B1], 
            model_dec = [decoder_T1B1, decoder_T1B1], 
            discriminator = [disc_T1, disc_B1],
            optimizer = optimizer, 
            optimizer_disc = optimizer_disc,
            criterion = [criterion_cyc, criterion_GAN], 
            batch = batch,
            epoch = e, 
            device = device, 
            writers = [writer_train, writer_test],
            model_path_save = model_path_save,
            comment = comment,
            lambda_normal = lambda_normal, 
            lambda_cyc = lambda_cyc,
            mask_roi = mask_roi, 
            disc = disc)
    
    print('\n'*3)

    if (e)%10 == 0:
        preview_results(test_data_dir, [encoder_T1, decoder_T1B1], e, device, path_test_save, 
                    pad=pad, mask_roi=mask_roi)
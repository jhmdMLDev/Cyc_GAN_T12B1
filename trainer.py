from genericpath import isfile
import torch
import nibabel as nib
import numpy as np
import os
from torch.autograd import Variable
from Data_loader import preview_dataloader
from utils import save_input

def preview_results(test_data_dir, model, e, device, path_test_save, pad, mask_roi=True):
    with torch.no_grad():
        T1, B1, affine, mx_T1 = preview_dataloader(test_data_dir, pad, mask_roi)

        T1 = torch.unsqueeze(T1,0)

        enc = model[0]
        dec = model[1]
        enc = enc.to(device)
        dec = dec.to(device)
        T1 = T1.to(device)

        if e==0:
            save_input(T1, 'T1')

        latent = enc(T1)
        output = dec(latent)

        output = np.squeeze(output.detach().cpu().numpy())

        img = nib.Nifti1Image(output, np.eye(4))
        nib.save(img, os.path.join(path_test_save, 'B1_pred_{}.nii.gz'.format(e)))  
        img = nib.Nifti1Image(B1 - output, np.eye(4))
        nib.save(img, os.path.join(path_test_save, 'B1_diff_{}.nii.gz'.format(e)))  
        if not os.path.isfile(os.path.join(path_test_save, 'B1_test.nii.gz')):
            img = nib.Nifti1Image(B1, np.eye(4))
            nib.save(img, os.path.join(path_test_save, 'B1_test.nii.gz'))  
    return


def train(dataloaders, model_enc, model_dec, discriminator, optimizer, optimizer_disc, 
            criterion, batch, epoch, device, phase=['train', 'test'], disc=False, writers=None,
            model_path_save=None, comment=None, checkpoint_interval=10,
            lambda_normal=0, lambda_cyc=10, mask_roi=True):

    def writer_saver(writer):
        writer.add_scalar('Loss', running_loss, epoch)
        writer.add_scalar('Loss T1', running_loss_T1, epoch)
        writer.add_scalar('Loss B1', running_loss_B1, epoch)
        writer.add_scalar('Loss gen', running_loss_gen, epoch)
        writer.add_scalar('Loss disc', running_loss_disc, epoch)
        writer.add_scalar('Loss T1 cyc', running_loss_T1_cyc, epoch)
        writer.add_scalar('Loss B1 cyc', running_loss_B1_cyc, epoch)
        return

    def model_saver():
        ####  .module
        torch.save(enc_T1.state_dict(), model_path_save + "/ENC_T1_{}_{}.model".format(comment, epoch))
        torch.save(enc_B1.state_dict(), model_path_save + "/ENC_B1_{}_{}.model".format(comment, epoch))
        torch.save(dec_T1B1.state_dict(), model_path_save + "/DEC_T1B1_{}_{}.model".format(comment, epoch))
        torch.save(dec_B1T1.state_dict(), model_path_save + "/DEC_B1T1_{}_{}.model".format(comment, epoch))
        torch.save(disc_T1.state_dict(), model_path_save + "/DISC_T1_{}_{}.model".format(comment, epoch))
        torch.save(disc_B1.state_dict(), model_path_save + "/DISC_B1_{}_{}.model".format(comment, epoch))
        return

    print('Epoch {}'.format(epoch))

    target_real = Variable(torch.Tensor(batch).fill_(1.0), requires_grad=False).to(device)
    target_fake = Variable(torch.Tensor(batch).fill_(0.0), requires_grad=False).to(device)

    enc_T1 = model_enc[0].to(device)
    enc_B1 = model_enc[1].to(device)
    dec_T1B1 = model_dec[0].to(device)
    dec_B1T1 = model_dec[1].to(device)
    disc_T1 = discriminator[0].to(device)
    disc_B1 = discriminator[1].to(device)
    for ph in phase:
        if ph=='train':
            enc_T1.train()
            enc_B1.train()
            dec_T1B1.train()
            dec_B1T1.train()
            disc_T1.train()
            disc_B1.train()
            dataloader = dataloaders[0]
            writer = writers[0]
        
        elif ph=='test':
            enc_T1.eval()
            enc_B1.eval()
            dec_T1B1.eval()
            dec_B1T1.eval()
            disc_T1.eval()
            disc_B1.eval()
            dataloader = dataloaders[1]
            writer = writers[0]

        running_loss = 0
        running_loss_T1 = 0
        running_loss_B1 = 0
        running_loss_gen = 0
        running_loss_disc = 0
        running_loss_T1_cyc = 0
        running_loss_B1_cyc = 0
        with torch.set_grad_enabled(ph=='train'):
            for i, (T1, B1, brain) in enumerate(dataloader):
                optimizer.zero_grad()
                optimizer_disc.zero_grad()

                if epoch==0 and i==0:
                    save_input(T1, path=model_path_save+'/T1')
                    save_input(B1, path=model_path_save+'/B1')

                T1 = T1.to(device)
                B1 = B1.to(device)
                brain = brain.to(device)
                
                # Normal path
                latent_T1 = enc_T1(T1)
                output_B1_fake = dec_T1B1(latent_T1) * brain
                latent_B1 = enc_B1(B1)
                output_T1_fake = dec_B1T1(latent_B1) * brain
                # Cyc path
                output_B1_fake_in = torch.cat((output_B1_fake, B1[:,1:,:,:]), 1) 
                output_T1_fake_in = torch.cat((output_B1_fake, T1[:,1:,:,:]), 1) 

                latentT1B1T1 = enc_B1(output_B1_fake_in)
                output_T1B1T1 = dec_B1T1(latentT1B1T1) * brain
                latentB1T1B1 = enc_T1(output_T1_fake_in)
                output_B1T1B1 = dec_T1B1(latentB1T1B1) * brain
                #save_input(output_B1_fake_in, 'output_B1_fake_in')
                # GAN path
                input_T1_fake = torch.cat((output_T1_fake, B1[:,0:1,:,:]), 1)
                input_B1_fake = torch.cat((output_B1_fake, T1[:,0:1,:,:]), 1)
                input_T1_real = torch.cat((T1[:,0:1,:,:], B1[:,0:1,:,:]), 1)
                input_B1_real = torch.cat((B1[:,0:1,:,:], T1[:,0:1,:,:]), 1)
                p_T1_fake = disc_T1(input_T1_fake)
                p_B1_fake = disc_B1(input_B1_fake)
                p_T1_real = disc_T1(input_T1_real)
                p_B1_real = disc_B1(input_B1_real)

                # Loss Calculations
                loss_disc = criterion[1](p_T1_fake, target_fake) + criterion[1](p_B1_fake, target_fake) + \
                                    criterion[1](p_T1_real, target_real) + criterion[1](p_B1_real, target_real)
                loss_disc = loss_disc * 0.5
                loss_gen = criterion[1](p_T1_fake, target_real) + criterion[1](p_B1_fake, target_real)

                loss_T1 = criterion[0](T1, output_T1_fake)
                loss_B1 = criterion[0](B1, output_B1_fake)
                loss_T1_cyc = criterion[0](T1, output_T1B1T1)
                loss_B1_cyc = criterion[0](B1, output_B1T1B1)

                loss = loss_gen + \
                    loss_T1 * lambda_normal + loss_B1 * lambda_normal + \
                    loss_T1_cyc * lambda_cyc + loss_B1_cyc * lambda_cyc

                if ph=='train' and not disc:
                    loss.backward()
                    optimizer.step()
                elif ph=='train' and disc:
                    loss_disc.backward()
                    optimizer_disc.step()

                running_loss += loss.item()
                running_loss_T1 += loss_T1.item()
                running_loss_B1 += loss_B1.item()
                running_loss_gen += loss_gen.item()
                running_loss_disc += loss_disc.item()
                running_loss_T1_cyc += loss_T1_cyc.item()
                running_loss_B1_cyc += loss_B1_cyc.item()

                if ph == "train" and ((i + 0) % 60) == 0:
                    print("minibatch {} of {}:"" runnning_loss : {:.4f}, ".format(
                            i, len(dataloader), running_loss / (i + 1)))

                torch.cuda.empty_cache()

        # Save and print checkpoint
        if not disc:
            writer_saver(writer)
            print('Loss {} (epoch {}): {:.4f}'.format(ph, epoch, running_loss))
        # else:
        #     print('Loss Discriminator {} (epoch {}): {:.4f}'.format(phase, epoch, running_loss_disc))


    if (epoch + 0) % checkpoint_interval==0 and not disc:
        model_saver()

    return

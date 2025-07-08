# Cyc_GAN_T12B1

This project implements a CycleGAN-based deep learning model for domain transfer in MRI imaging. Specifically, it translates between B1 maps and T2-weighted MRI images.

## Overview

The goal is to input MRI B1 images and generate the corresponding T2-weighted images (or vice versa) using a CycleGAN framework. This approach is helpful in cases where acquiring one modality is easier or more feasible than the other.

## Method

A CycleGAN model is used to perform unpaired image-to-image translation between B1 and T2 MRI domains. The architecture includes two generators and two discriminators, enabling bidirectional translation.

- **Generators**: Encoder-decoder architectures for both B1→T2 and T2→B1 translations.
- **Discriminators**: Separate discriminators for each domain.
- **Loss Functions**: Combination of L1 and L2 norms, along with adversarial and cycle consistency losses.
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum.

## Getting Started

1. Update the paths in the configs.py file to point to your dataset directories.
2. To train the model:
   - On Linux-based systems, run the setup script:  
     ```bash
     ./cc_setup.sh
     ```
   - Or run the training script directly:  
     ```bash
     python Train.py
     ```

## Notes

- This model assumes paired training data.
- Make sure your datasets are preprocessed and formatted correctly.

## License

[Add license information here if applicable.]


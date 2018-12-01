# Generating Steganographic Images via Adversarial Training

This is a toy example (on CelebA) of how to run experiments presented in https://arxiv.org/abs/1703.00371

## To run

1. Download CelebA dataset and save to `./data`
2. Run `python main.py`

## Results 

Example of real images. 

<img src="results/real_output_output_32.png"  width="300" height="240">

Example of Alice generated images at Epoch 0. 

<img src="results/noise_output_output_0.png"  width="300" height="240">

Example of Alice generated images at Epoch 500. 

<img src="results/noise_output_output_500.png"  width="300" height="240">

Loss plots.

<img src="results/eve_loss.png"  width="300" height="240">

Bob decoding success.

<img src="results/correct_bits.png"  width="300" height="240">

## Notes

Hyperparameter optimization is needed for optimal results (and other tweaks like using high capacity networks).

 # DAE with Noise added at Different Layers
This repository is for comparing the performance of Denoising Autoencoder (DAE) when noise is added to different layers.

# Usage

```shell
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
conda install matplotlib
```

```shell
    python train.py
    python plot_result.py
```
# Symstem model
Each index(noise_idx) represents the location where noise is added.
- train SNR : 12dB
- test SNR : 12dB
- Dataset : Mnist (train : 54000, test : 6000)
<img src="result_img/system model.png" width=1080px>


# 2-dim Manifold result

*noise_idx* | *result* | *noise_idx* | *result*
:---: | :---: |:-----------:| :---: | 
0 |<img src="result_img/autoencoder_2023_07_28_14_21_05_0_manifold_idx_0.png" width=280px> |      2      | <img src="result_img/autoencoder_2023_07_28_14_25_12_2_manifold_idx_2.png" width=280px> 
4 |<img src="result_img/autoencoder_2023_07_28_14_34_13_4_manifold_idx_4.png" width=280px> |      6      | <img src="result_img/autoencoder_2023_07_28_14_45_04_6_manifold_idx_6.png" width=280px> 
8 |<img src="result_img/autoencoder_2023_07_28_14_55_46_8_manifold_idx_8.png" width=280px> |     10      | <img src="result_img/autoencoder_2023_07_28_15_05_15_10_manifold_idx_10.png" width=280px> 

# re-generation result
*noise_idx* |                                        *result*                                        
:---: |:--------------------------------------------------------------------------------------:
0 | <img src="result_img/autoencoder_2023_07_28_14_21_05_0_result_idx_0.png" width=280px>  | 
1 | <img src="result_img/autoencoder_2023_07_28_14_25_12_2_result_idx_2.png" width=280px>  | 
2 | <img src="result_img/autoencoder_2023_07_28_14_34_13_4_result_idx_4.png" width=280px>  | 
3 | <img src="result_img/autoencoder_2023_07_28_14_45_04_6_result_idx_6.png" width=280px>  | 
4 | <img src="result_img/autoencoder_2023_07_28_14_55_46_8_result_idx_8.png" width=280px>  | 
5 | <img src="result_img/autoencoder_2023_07_28_15_05_15_10_result_idx_10.png" width=280px> | 

# re-generation with noisy-input result 
*DNN* |                                    *CNN*                                     | *DAE* 
:---: |:----------------------------------------------------------------------------:| :---: 
<img src="result_img/3_dnn_autoencoder_2023_07_26_14_55_06.png" width=280px> | <img src="result_img/3_cnn_autoencoder_2023_07_26_17_01_53.png" width=280px> | <img src="result_img/3_DAE_autoencoder_2023_07_26_18_00_02.png" width=280px>


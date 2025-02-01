# RETFusion

RETFusion is a foundational model designed for efficient pre-training on mixed retinal images from similar modalities. This project investigates the hypothesis that mixing medical images from similar modalities can serve as an efficient data augmentation strategy, reducing data and computational requirements while achieving competitive performance. 

## 🔗 Acknowledgment

We sincerely appreciate the RETFound project and its contributors for their foundational work in retinal imaging with self-supervised learning. RETFusion builds upon RETFound's pre-trained models and methodologies. 

Please check out the official RETFound repository here: [RETFound GitHub](https://github.com/rmaphoh/RETFound_MAE)

If you find RETFusion helpful, please consider citing our work and the RETFound paper:

```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## 🚀 Features

- Efficient pre-training using a mixture of retinal images from similar modalities.
- Demonstrates competitive performance with reduced computational resources.
- Evaluated on clinically relevant applications, including diabetic retinopathy detection.

## 🔥 RETFusion Checkpoint

To use the pre-trained RETFusion model, download the checkpoint here:

[RETFusion Checkpoint](https://drive.google.com/file/d/1J1t7SMG3A13Hg7622mqNBFc251lMmY1p/view?usp=sharing)

## 🚀 Google Colab Notebook

To facilitate easy access and experimentation, we provide a Google Colab notebook for running RETFusion without the need for local installation. You can access it here:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P-bn-04LQ8p3Ru0gcFxlLJsTOfrKBRNi?usp=sharing)

## 🛠 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/RETFusion.git
    cd RETFusion
    ```
2. Create a Conda environment:
    ```bash
    conda create -n retfusion python=3.7.5 -y
    conda activate retfusion
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

Organize your dataset into the following structure:

```
├── data folder
    ├── train
        ├── class_a
        ├── class_b
    ├── val
        ├── class_a
        ├── class_b
    ├── test
        ├── class_a
        ├── class_b
```

## 📊 Data Availability

We provide data splits and model checkpoints to facilitate model comparison. Please remember to adjust `data_path`, `task`, and `nb_classes` for model fine-tuning and evaluation.

### Data Split

| Dataset  | Download Link 1  | Download Link 2 |
|----------|-----------------|-----------------|
| APTOS2019 | [Google Drive](https://drive.google.com/file/d/162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1uR8uUAnkO19lVT3beZuoMg) code:a2wg |
| MESSIDOR2 | [Google Drive](https://drive.google.com/file/d/1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1uR8uUAnkO19lVT3beZuoMg) code:a2wg |
| IDRID | [Google Drive](https://drive.google.com/file/d/1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1uR8uUAnkO19lVT3beZuoMg) code:a2wg |

### Official Websites

- APTOS2019: [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- MESSIDOR2: [ADCIS](https://www.adcis.net/en/third-party/messidor2/)
- IDRID: [IEEE Dataport](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

## 🔧 Fine-tuning RETFusion

To fine-tune the model on your dataset, run:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./your_dataset/ \
    --task ./finetune_your_task/ \
    --finetune ./RETFound_cfp_weights.pth \
    --input_size 224
```

For evaluation:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --eval --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --resume ./finetune_your_task/checkpoint-best.pth \
    --input_size 224
```



## 📝 Citation

If you use RETFusion in your work, please consider citing our paper alongside RETFound. 

We hope RETFusion contributes to advancing research in efficient foundation model development for medical imaging!
# RETFusion

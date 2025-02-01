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

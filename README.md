# DINO-Vision: Image classification and "what the model sees" with frozen DINOv2

Meta's **DINOv2** is a Vision Transformer trained with **self-supervised learning**: it learned from \~142M images without a single human label. This project shows how far those features go on their own:

1. **Classification with almost no training.** DINOv2 stays frozen; only a logistic-regression layer is trained on top. Evaluated on the 37-breed Oxford-IIIT Pets dataset (works on CIFAR-10 or your own photos too).
2. **Unsupervised part discovery.** PCA on DINOv2's per-patch features separates objects from background and colors matching parts (ears, eyes, legs) the same across different images, with no labels or training at all.

## Results

|Method (DINOv2 ViT-S/14, frozen)|Dataset|Accuracy|Macro-F1|
|-|-|-|-|
|k-NN (k=10, cosine)|Oxford-IIIT Pets|*93.16%*|*93.03%*|
|Linear probe|Oxford-IIIT Pets|*95.09%*|*95.0%*|

*(numbers come from `results/metrics.json`)*

**Patch-feature PCA** (top: input, bottom: first 3 principal components of the foreground patches as RGB)

!\[patch PCA](results/patch\_pca.png)

|t-SNE of embeddings|Confusion matrix|
|-|-|
|!\[t-SNE](results/tsne.png)|!\[confusion matrix](results/confusion\_matrix.png)|

## How it works

```
image ──► DINOv2 (frozen) ──► 384-dim CLS embedding ──► logistic regression ──► class
               │
               └──► one feature per 14×14 patch ──► PCA ──► RGB "what the model sees" map
```

|File|What it does|
|-|-|
|`dino\_utils.py`|Loads DINOv2 from `torch.hub`, preprocessing, feature extraction|
|`embed.py`|Runs every image through the frozen model and saves one embedding per image|
|`train\_probe.py`|Trains k-NN and a linear probe (the two standard ways DINO features are evaluated); saves metrics, confusion matrix, t-SNE|
|`patch\_pca.py`|Two-stage PCA on patch tokens: component 1 = foreground mask, then 3 components on foreground = RGB|
|`predict.py`|Classify new images with the trained probe|

Because only a linear layer is trained, the full pipeline takes minutes.

## Run it

```bash
git clone https://github.com/<your-username>/dino-vision.git
cd dino-vision
pip install -r requirements.txt

python embed.py --dataset pets                 # downloads Pets + DINOv2 weights (\~800 MB) on first run
python train\_probe.py
python patch\_pca.py data/oxford-iiit-pet/images/beagle\_1\*.jpg --max-images 4
python predict.py data/oxford-iiit-pet/images/pug\_10.jpg
```

Other options:

* `python embed.py --dataset cifar10 --max-train 5000` for CIFAR-10 (subset for speed on CPU)
* `python embed.py --dataset folder --data my\_photos` with your own images in `my\_photos/<class\_name>/\*.jpg`
* `--model base` or `--model large` for a bigger backbone (slower, more accurate)

No GPU? Use Google Colab (`Runtime → Change runtime type → T4 GPU`) and run the same commands with `!` in front.

## Next steps

* Compare against **DINOv3** (Aug 2025) and a supervised ImageNet ResNet-50 baseline
* Few-shot: how accuracy changes with 1, 5, 10 labeled images per class
* Train a small segmentation head on the patch tokens

## References

* Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021
* Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
* Siméoni et al., *DINOv3*, 2025


# A Novel Alzheimer’s Disease Classification Framework Integrating 3D CNN with Bidirectional Feature Pyramid Network and Vision Transformers

## Abstract
Alzheimer’s disease (AD) is a progressive neurodegenerative disorder that severely affects memory and cognitive function, underscoring the need for accurate and early diagnostic tools. Deep learning has shown strong potential in automated AD detection from brain MRI scans; however, many existing approaches either overlook the full volumetric context of MRI data or rely on limited feature fusion strategies that weaken spatial coherence. To address these challenges, this study proposes BiFPN3DViT, a hybrid deep learning framework that integrates a hierarchical 3D convolutional backbone as the foundational component of a Bidirectional Feature Pyramid Network (BiFPN), coupled with a Vision Transformer (ViT) for global contextual learning. The unified 3D CNN–BiFPN module jointly extracts and fuses multi-scale volumetric features through both bottom-up and top-down pathways, forming a rich spatial representation that is subsequently refined by the transformer encoder. Evaluated on 4,706 MRI scans from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset, the proposed model achieved over 92\% classification accuracy across AD, mild cognitive impairment (MCI), and cognitively normal (CN) subjects. Comparative and ablation studies confirm the complementary strengths of convolutional and transformer-based modeling, while attention visualization highlights clinically relevant brain regions. These results demonstrate that BiFPN3DViT offers a scalable, interpretable, and computationally efficient framework for MRI-based Alzheimer’s disease diagnosis, advancing the integration of multi-scale feature learning and attention-driven analysis in neuroimaging.

## Preprocessing Pipeline
![Model Architecture](images/preprocessing_pipeline.png)

## Model Architecture
![Model Architecture](images/BiFPN_block.png)
![Model Architecture](images/transformer_block.png)
## Methodology
![Mmethodology](images/architecture_overview11.png)


## Important Preprocessing

### HD-BET
This approach is utilized for skull stripping from the images. Preprocess data using the [HD-BET](https://github.com/MIC-DKFZ/HD-BET) method and prepare the data as instructed below.

![Model Architecture](images/MRI_before_processing.png)
![Model Architecture](images/preprocessing_samples.jpeg)

## Datset Prepration:
Your dataset should be structure in a DataFrame.
```
| ADNI_path | Group |
|-----------|-------|


```
## Requirements
- Python 3.9.19
- ```sh
conda create MAF python=3.9.19
```

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/...
    ```
2. Navigate to the project directory:
    ```sh
    cd A...
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model, run the following command:
```sh
python train.py
```

### Evaluation
To evaluate the model, run the following command:
```sh
python evaluate.py
```

### GradCAM Results
<p align="center">
    <img src="images/GradCAM1.png" alt="GradCAM Result 1" width="45%">
    <img src="images/GradCAM2.png" alt="GradCAM Result 2" width="45%">
</p>


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


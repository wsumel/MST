# mst
The official implementation for the **ACM MM 2025** paper [Multi-State Tracker: Enhancing Efficient Object Tracking via Multi-State Specialization and Interaction](https://arxiv.org/pdf/2508.11531).

[[Raw Results]()]

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=joint-feature-learning-and-relation-modeling)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=joint-feature-learning-and-relation-modeling)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-trackingnet&#41;]&#40;https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=joint-feature-learning-and-relation-modeling&#41;)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-uav123)](https://paperswithcode.com/sota/visual-object-tracking-on-uav123?p=joint-feature-learning-and-relation-modeling)
 -->
<p align="center">
  <img width="85%" src="https://github.com/wsumel/MST/tree/main/assets/method.png" alt="Framework"/>
</p>

## News
**[Dec. 12, 2025]**
- MST was accepted by ACM MM2025.


## Highlights

### :star2: New Efficient Tracking Framework
MST is a simple, neat, high-performance **Efficient tracking framework**.
MST achieves good performance on multiple benchmarks. MST can serve as a strong baseline for further research.

| Tracker     | GOT-10K (AO) | LaSOT (AUC) | TrackingNet (AUC) | UAV123(AUC) |
|:-----------:|:------------:|:-----------:|:-----------------:|:-----------:|
| MST | 69.6         | 65.8        | 81.0              | 68.4        |



### :star2: Contribution
- **Multi-State Tracking Architecture:** We introduce the Multi-State Tracker (MST), which leverages multiple state representations to enhance tracking accuracy and robustness, enabling better handling of variations in appearance, occlusions, and motion blur. 
- **Key Technical Innovations:** We propose three key modules Multi-State Generation (MSG), State-Specific Enhancement (SSE), and Cross-State Interaction (CSI), with the latter two built upon the Hidden State Adaptation-based State Space Duality (HSA-SSD). Together, these modules enable the generation of diverse state-aware features, the refinement of individual state representations, and effective information exchange across states. Importantly, they enhance tracking performance while introducing only minimal computational overhead. 
- **State-of-the-Art Performance:** Leveraging its innovative architectural design and key advancements, MST not only maintains exceptional processing speed but also delivers strong tracking performance, as demonstrated by extensive experimental evaluations across several benchmark datasets. 

### :star2: Good performance-speed trade-off

[//]: # (![speed_vs_performance]&#40;https://github.com/wsumel/MST/tree/main/assets/performence.png&#41;)
<p align="center">
  <img width="70%" src="https://github.com/wsumel/MST/tree/main/assets/performence.png" alt="speed_vs_performance"/>
</p>

## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n mst python=3.8
conda activate mst
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f mst_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [MAE ViT-Tiny weights](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE-Lite](https://github.com/wangsr126/mae-lite?tab=readme-ov-file) for more details).

```
python tracking/train.py --script mst --config vitt_256_mae_32x4_ep100_got --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/mst`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
<!-- Download the model weights from [Google Drive]()  -->

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/mst`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py mst vitt_256_mae_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py mst vitt_256_mae_32x4_ep100_got --dataset got10k_test --threads 16 --num_gpus 4

```
- TrackingNet
```
python tracking/test.py mst vitt_256_mae_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name mst --cfg_name vitt_256_mae_32x4_ep300
```



## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack), [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.  


## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{wang2025mst,
  title={Multi-State Tracker: Enhancing Efficient Object Tracking via Multi-State Specialization and Interaction},
  author={Shilei Wang and Gong Cheng and Pujian Lai and Dong Gao and Junwei Han},
  booktitle={ACM MM},
  year={2025}
}
```

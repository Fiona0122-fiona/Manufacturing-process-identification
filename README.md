# Manufacturing process identification from 3D point cloud models using semantic segmentation

## CAD Dataset:

<img width="2495" height="913" alt="image" src="https://github.com/user-attachments/assets/16380356-bdeb-4d6c-96ae-63dbf9f8a117" />

## MRIConv++ model:

<img width="3493" height="1586" alt="image" src="https://github.com/user-attachments/assets/e4e1e319-b565-4eb1-83fa-d749419927dd" />

## PRIConv++ model:

<img width="3493" height="1091" alt="image" src="https://github.com/user-attachments/assets/6602418a-7022-4783-b6b3-bc568ed01d3f" />


Journal of manufacturing systems, 2025

If you found this paper useful in your research, please cite:
```
@article{Liu2025ManufacturingPI,
  title={Manufacturing process identification from 3D point cloud models using semantic segmentation},
  author={Xiaofang Liu and Zhichao Wang and Shreyes N. Melkote and David W. Rosen},
  journal={Journal of Manufacturing Systems},
  year={2025},
  url= {https://doi.org/10.1016/j.jmsy.2025.07.023}
}
```


## Installation
This repo provides the MRIConv++ and PRIConv++ source codes, which had been tested with Python 3.9.7, PyTorch 1.9.0, CUDA 11.1 on Ubuntu 20.04. Our codes are implemented based on Zhi Yuan Zhang's Pytorch implementation of [ RIConv++(https://github.com/cszyzhang/riconv2)], Xu Yan's [PointNet++(Pytorch)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and ShaoShuai Shi's [PointNet++ cuda operations](https://github.com/sshaoshuai/Pointnet2.PyTorch).  

Install the pointnet++ cuda operation library by running the following command:
```
cd models/pointnet2/
python setup.py install
cd ../../
```

## Usage


### Segmentation
We perform semantic segmentation on mriconv2_3d and priconv2_3d respectively.

```
cd data_utils
python collect_indoor3d_data.py
```

Processed data will save in `../data/s3dis/mriconv2_3d/`. (**Note**: the `data/` folder is outside the project folder)
Processed data will save in `../data/s3dis/priconv2_3d/`. (**Note**: the `data/` folder is outside the project folder)
The raw data is also provided in the folder "Raw Data".

Training:

```
python3 train_semseg.py
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** [18.2MB](log/sem_seg/pretrained) directly:

```
python3 test_semseg.py
```


## License
This repository is released under MIT License (see LICENSE file for details).

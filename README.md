# MHST-Net
This an official Pytorch implementation of our paper ["MHST: Multiscale Head Selection Transformer for Hyperspectral and LiDAR Classification"]().The specific details of the model are as follows.
![MHST-Net](./figure/MHST-Net.png)
****
# Datasets
- [Houston2013 dataset](https://hyperspectral.ee.uh.edu/?page_id=459)
, initially featured in the 2013 IEEE GRSS data fusion contest, was captured using the ITRES CASI-1500 sensor over the University of Houston campus in June 2012. The dataset includes a hyperspectral image (HSI) and a LiDAR-based digital surface model (DSM), both sharing dimensions of 349 × 1905 pixels and a spatial resolution of 2.5 m. The hyperspectral image comprises 144 spectral bands covering a wavelength range from 0.364 to 1.046 μm. The dataset encompasses 15,029 ground-truth samples across 15 classes, making it a valuable resource for various geospatial analyses and land cover mapping applications.
- [Trento](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)
****
# Train MHST-Net
``` 
python demo.py
``` 
****
# Results
All the results presented here are referenced from the original paper.
| Dataset | OA (%) | AA (%) | Kappa (%) |
| :----: |:------:|:------:|:---------:|
| Houston  | 96.19  | 96.80  |   95.88   |
| Trento  | 99.45  | 99.09  |   99.26   |
****
# Citation
If you find this paper useful, please cite:
``` 
@ARTICLE{,
  author={Kang Ni, Duo Wang, Zhizhong Zheng},
  journal={}, 
  title={}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}
}
```
****
# Contact
Duo Wang: [b21041510@njupt.edu.cn](b21041510@njupt.edu.cn)

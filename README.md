# MHST-Net
This an official Pytorch implementation of our paper ["MHST: Multiscale Head Selection Transformer for Hyperspectral and LiDAR Classification"]().The specific details of the model are as follows.
![MHST-Net](./figure/MHST-Net.png)
****
# Datasets
- [The Houston2013 dataset](https://hyperspectral.ee.uh.edu/?page_id=459)
includes a hyperspectral image (HSI) and a LiDAR-based digital surface model (DSM), collected by the National Center for Airborne Laser Mapping (NCALM) using the ITRES CASI-1500 sensor over the University of Houston campus in June 2012. The HSI comprise 144 spectral bands covering a wavelength range from 0.38 to 1.05 µm while LiDAR data are provided for a single band. Both the HSI and LiDAR data share dimensions of 349 × 1905 pixels with a spatial resolution of 2.5 m. The dataset contains 15 categories, with a total of 15,029 real samples available. 
- [The Trento dataset](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)
comprises HSI and LiDAR data obtained from southern Trento, Italy. The HSI was collected by an AISA Eagle sensor, consisting of 63 spectral bands with a wavelength range from 0.42 to 0.99 µm. LiDAR data with 1 raster were acquired by the Optech ALTM 3100EA sensor. The scene consists of 166 × 600 pixels, with a spatial resolution of 1 m. This dataset contains 6 land cover types
with a total of 30,214 real samples. 
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

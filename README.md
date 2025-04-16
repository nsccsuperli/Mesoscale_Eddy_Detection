
### Resources
- Implementation of **MFDE** (Mesoscale eddy detection based on multi-feature fusion network) 

### Datasets 
1. SSH data: [ADT Dataset ](https://data.marine.copernicus.eu/products))

2. SST:  [[SST Dataset ](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html)]

3. Our data(SSH,SST and Contour data): [[Google Drive](https://drive.google.com/drive/folders/15RsEpo9WsvZYQ0KB756dg5G4ORsY27Eg?usp=drive_link)]

4. Best_model:[[Google Drive](https://drive.google.com/file/d/1xQQEq1BaOtihsUWqsV3XLQ31EhoVM8Ep/view?usp=drive_link)]
### Usage
- Download the data, setup data-paths in the datasets
- Use the training scripts[train.py] for paired training of MFED


| The Overall Flowchart of the Proposed Architecture |
|:---------------------------------------------------|
| ![det-enh](./data/Fig1.png)                        |

| Mesoscale eddy semantic segmentation task |
|:------------------------------------------|
| ![det-enh](./data/Fig8.png)               |

| Mesoscale eddy contours detection task |
|:---------------------------------------|
| ![det-enh](./data/Fig9.png)            |


## Reference
If you find this repository useful or our work is related to your research, please kindly cite it:
```
@article{HUO2024103714,
title = {High kinetic energy mesoscale eddy identification based on multi-task learning and multi-source data},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {128},
pages = {103714},
year = {2024},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2024.103714},
url = {https://www.sciencedirect.com/science/article/pii/S1569843224000682},
author = {Jidong Huo and Jie Zhang and Jungang Yang and Chuantao Li and Guangliang Liu and Wei Cui},
keywords = {Mesoscale eddy, Deep learning, Contour detection, Multi-source features},
}
```


### Acknowledgements


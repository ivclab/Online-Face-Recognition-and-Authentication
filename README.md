# Evalution Code of Face Recognition and Authentication with Online Registration
Official implementation of [Data-specific Adaptive Threshold for Face Recognition and Authentication](https://arxiv.org/abs/1810.11160).

Created by [Hsin-Rung Chou](https://github.com/Sherry40931), [Jia-Hong Lee](https://github.com/Jia-HongHenryLee), [Yi-Ming Chan](https://github.com/yimingchan), Chu-Song Chen.

The code is released for academic research use only. For commercial use, please contact [Dr. Chu-Song Chen](https://www.iis.sinica.edu.tw/pages/song/)(song@iis.sinica.edu.tw).

## Introduction
Many face recognition systems boost the performance using deep learning models, but only a few researches go into the mechanisms for dealing with online registration. Although we can obtain discriminative facial features through the state-of-the-art deep model training, how to decide the best threshold for practical use remains a challenge. We develop a technique of adaptive threshold mechanism to improve the recognition accuracy. We also design a face recognition system along with the registering procedure to handle online registration. Furthermore, we introduce a new evaluation protocol to better evaluate the performance of an algorithm for real-world scenarios. Under our proposed protocol, our method can achieve a 22\% accuracy improvement on the LFW dataset.

## Prerequisition
```bash
$ pip install -r requirement.txt
```

## Usage

### 1. Dump embedding
#### - Use FaceNet (reproduce result in paper)
We used [FaceNet](https://github.com/davidsandberg/facenet) model version [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) to generate embedding of color_FERET, LFW and Adience in our experiments. You can find the embeddings under the data repository. We shuffle each dataset for 10 times so that the register orders are different. Then we compute the average accuracy from these 10 experiments.

1. Download [FaceNet model 20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit)
2. ```$ python dump_embeddings.py --model 20170512-110547/ [data_dir]```
Please follow the [instruction](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images) to set up your dataset repository.


#### - Use your own deep learning model
1. Generate features with fixed dimension (e.g 128 or 256) with your chosen model and save them into a csv file with the following format:
```bash
   [image_name], [features], [threshold(initial value = 0)], [path_to_image]
```
2. You can check the csv file under the data repository for reference

### 2. Run simulation and evaluation with adaptive threshold (with color_FERET)
- Run simulation:
If ```max_compare_num``` is less than 1, the program will compare the registering embedding with all the embeddings sotred in the simulated database. (It will take a lot of time)
```bash
$ python simulator_v4_adaptive_thd.py data/color_FERET --max_compare_num 100
```

- Compute average accuracy from 10 experiments
```bash
$ python get_avg_accuracy.py result/Simulator_v4_features_color_FERET_v
```

### 3. Run simulation and evaluation with fixed threshold
```bash
$ python simulator_v4_fixed_thd.py data/color_FERET 0.39
$ python get_avg_accuracy.py result/Simulator_v4_features_color_FERET_v
```


## Result
<!-- <img align="center" src="https://i.imgur.com/GLOBBam.png"> -->
![](https://i.imgur.com/GLOBBam.png)


## Reference Resource
- [FaceNet](https://github.com/davidsandberg/facenet)
- [color FERET](https://www.nist.gov/itl/products-and-services/color-feret-database)
- [LFW](http://vis-www.cs.umass.edu/lfw/)
- [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender)

## Citation
Please cite following paper if these codes help your research:

    @inproceedings{chou2019data,
    title={Data-specific Adaptive Threshold for Face Recognition and Authentication},
    author={Chou, Hsin-Rung and Lee, Jia-Hong and Chan, Yi-Ming and Chen, Chu-Song},
    booktitle={2019 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)},
    pages={153--156},
    year={2019},
    organization={IEEE}
    }

## Contact
Please feel free to leave suggestions or comments to [Hsin-Rung Chou](https://github.com/Sherry40931), [Jia-Hong Lee](https://github.com/Jia-HongHenryLee), [Yi-Ming Chan](https://github.com/yimingchan), [Chu-Song Chen](https://www.iis.sinica.edu.tw/pages/song/)(song@iis.sinica.edu.tw)
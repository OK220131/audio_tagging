# audio_tagging(PANNs inferece)

**panns_inference** provides an easy to use Python interface for audio tagging and sound event detection. The audio tagging and sound event detection models are trained from PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition: https://github.com/qiuqiangkong/audioset_tagging_cnn

## Installation
PyTorch>=1.0 is required.
```
$ pip install panns-inference
$ pip install numba==0.56.4
```
以下のダウンロードを行った後test.pyと同じ階層に配置

https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1
https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1


## Usage
```
$ python test.py
```
## Example
```
labeling
['test.py', '笑い声.mp3']
load Model
------ Audio tagging ------
Checkpoint path: Cnn14_mAP=0.431.pth
Using CPU.
Speech: 0.067
Laughter: 0.050
Baby laughter: 0.039
Belly laugh: 0.034
Inside, small room: 0.025
Snicker: 0.018
Child speech, kid speaking: 0.018
Giggle: 0.016
Chuckle, chortle: 0.013
Inside, large room or hall: 0.011
------ Sound event detection ------
Checkpoint path: Cnn14_DecisionLevelMax_mAP=0.385.pth
Using CPU.
idxes: [16, 17, 22, 23]
len: 3542
cry= 3.571428571428571 %,laugh= 96.42857142857143 %
Save fig to results/笑い声.mp3_result.png
```


## Cite
[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

# Representing Animatable Avatar via Factorized Neural Fields

Official Repository for SGP 2025 paper [*Representing Animatable Avatar via Factorized Neural Fields*](). 

## Getting Started
* Clone this repo: `git clone https://github.com/ChunjinSong/facavatar.git`
* Create a python virtual environment and activate. `conda create -n facavatar python=3.7` and `conda activate facavatar`
* Install dependenices. `cd facavatar`, `pip install -r requirement.txt` and `python setup.py develop`
* Download [SMPL model](https://smpl.is.tue.mpg.de/download.php) (1.0.0 for Python 2.7 (10 shape PCs)) and move them to the corresponding places:
```
mkdir lib/smpl/smpl_model/
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_FEMALE.pkl
mv /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_MALE.pkl
mv /path/to/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_NEUTRAL.pkl
```

## Training and Testing
We support to train and test the model on 2 GPUs. Please set `gender`, `data_dir`, `subject`, `img_scale_factor`, `bgcolor`, `test_type` in `./script/launch.sh` first, then run
```
bash ./script/launch.sh
```
The results can be found at `outputs/`.


## Data Preprocessing
We are unable to share the data we used due to licensing restrictions. However, we provide the data processing code for LS-Avatar and the baselines. Please refer to the link [here](https://github.com/ChunjinSong/human_data_processing).

## Acknowledgement
We have utilized code from several outstanding research works and sincerely thank the authors for their valuable discussions on the experiments, including those from [Vid2Avatar](https://github.com/MoyGcc/vid2avatar), [HumanNeRF](https://github.com/chungyiweng/humannerf), [SMPL-X](https://github.com/vchoutas/smplx), [MonoHuman](https://github.com/Yzmblog/MonoHuman), [PM-Avatar](https://github.com/ChunjinSong/pmavatar) and [NPC](https://github.com/LemonATsu/NPC-pytorch).

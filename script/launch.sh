gender='neutral'
data_dir='./data/zju_mocap/394_training.h5' # data path
subject='394' # subject
img_scale_factor=0.5 # scaling the input image
bgcolor=0.0   # 0--black 1--white
test_type='novel_view'  # 'novel_view' or 'novel_pose'

python train.py \
    dataset.metainfo.gender=${gender} \
    dataset.metainfo.data_dir=${data_dir} \
    dataset.metainfo.subject=${subject} \
    dataset.metainfo.img_scale_factor=${img_scale_factor} \
    dataset.metainfo.bgcolor=${bgcolor}

python test.py \
    dataset.metainfo.gender=${gender} \
    dataset.metainfo.data_dir=${data_dir} \
    dataset.metainfo.subject=${subject} \
    dataset.metainfo.img_scale_factor=${img_scale_factor} \
    dataset.metainfo.bgcolor=${bgcolor} \
    dataset.testing.type=${test_type} \



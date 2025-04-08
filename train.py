# from lib.libmise import mise
from v2a_model import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.run}")

    trainer = pl.Trainer(
        gpus=2,
        strategy='ddp',
        accelerator="gpu",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        max_steps=opt.model.max_step,
        check_val_every_n_epoch=opt.model.epoch_save,
        logger=logger,
        log_every_n_steps=100,
        num_sanity_val_steps=0
    )

    model = V2AModel(opt)
    trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
    validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)

    if opt.model.is_continue == True:
        checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else: 
        trainer.fit(model, trainset, validset)

if __name__ == '__main__':
    main()
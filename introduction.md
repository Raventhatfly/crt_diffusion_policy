# How to use ARX Diffusion Policy Repo

```
python demo_arx_robot.py -o "/home/philaptop/wfy/dataset"
```
```
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```
```
python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/pick_and_place_250320 task=real_arx dataloader.batch_size=16
```
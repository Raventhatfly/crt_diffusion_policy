# How to use ARX Diffusion Policy Repo

### Open the SLCAN port
```
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
sudo slcand -o -f -s8 /dev/arxcan1 can1 && sudo ifconfig can1 up
```
### Collecting Data
```
python demo_arx_robot_new.py -o "/home/philaptop/wfy/dataset"
```

### Train the model
```
python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/pick_and_place_250320 task=real_arx dataloader.batch_size=16 training.num_epochs=500
```
### Inference
```
python eval_arx_robot.py -i /home/philaptop/wfy/repos/crt_diffusion_policy/data/epoch=1000-train_loss=0.002.ckpt -o /home/philaptop/wfy/recordings
```
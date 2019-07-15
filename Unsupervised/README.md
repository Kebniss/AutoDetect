# Unsupervised part

This is a variant of Pix2Pix, forked from [this repo](https://github.com/jctestud/pix2pixHD/tree/video).

Source for this idea was [this blog post](https://medium.com/element-ai-research-lab/modern-recipes-for-anomaly-detection-52150641074f), combined with [this other blog post](https://medium.com/@jctestud/video-generation-with-pix2pix-aed5b1b69f57).

Here are the instructions on how to start train on AWS.

1. Link to sample fire video: https://vimeo.com/62740159
    - Click download
    - Right mouse click on Original -> copy link url
    - Run this command in terminal: `wget -O fire.mp4 copied_url`
2. To run cmd line stuff, we need to activate conda for the command line python.
    - **Important**: CMD starts in ~, your notebooks are in `/pix2pixHD`
    - `cd /pix2pixHD`
    - `conda env list` shows all envs
    - `conda activate pytorch_p36` is the one we want
    - If it says conda not found, do this:
        1. run `sudo dpkg-reconfigure dash` in terminal, answer no
        2. `echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc`
        3. `source ~/.bashrc`
    - Next up, intall ffmpeg. Use the `install_ffmpeg.sh` script.
        - run each command in terminal
3. You might have to install
    - pytorch: `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
    - dominate: `conda install -c conda-forge dominate`
4. To split the video in frames run `python3 extract_frames.py -video fire.mp4 -name fire_dataset -p2pdir . -width 1280 -height 736`
5. To train on GPU run `python3 train_video.py --name fire_project --dataroot ./datasets/fire_dataset/ --save_epoch_freq 1 --ngf 32 --gpu True`

To generate a video, use the following:
```
python3 generate_video.py --name <name of a trained model> --dataroot <path to where your images are> --fps 30 --ngf 32
```

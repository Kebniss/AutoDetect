# AutoDetect
Code for my Insight project, AutoDetect. This project is about finding anomalies in videos. 

This repo contains two approaches: supervised and unsupervised/self-supervised. The code for Supervised is pure PyTorch, whereas the Unsupervised codebase is a mix of numpy/pytorch for the baseline (simple Frame2Frame similarity) and piggybacks on Pix2Pix for the GAN.

Please check my [slide deck](https://docs.google.com/presentation/d/1RdilhTLWwx9OcFoxeFg5OQffbbGVYqci/edit#slide=id.p17) to see the results of each model.

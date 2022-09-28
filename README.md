# NMSG
Official implementation of paper "Self-Supervised Cryo-Electron Tomography Volumetric Image Denoising with Noise Modeling and Sparsity Guidance".
<br>
![Overall architecture](./Figure1.png)
<br>
Figure 1. Overall architecture of NMSG framework. The workflow started with noisy projections which are firstly reconstructed as raw noisy image. Then, noisy projections are filtered and reconstructed as over-smoothed image which is fed into the sparsity extractor to capture the sparsity information. The raw noisy image is filtered with 3D Gaussian filter to generate filtered image as guidance. The synthetic noise volumes are generated from noisy image using noise synthesizer. The synthetic noise volumes are added to raw noisy image to generate synthetic noisy input for the training.
<br>
![simulated data result](./Figure5.png)
<br>
Figure 5. Visualized results of the simluated dataset with noise intensity of sigma=20. All images are selected from the center slices on x-, y- and z- axis.
<br>
![real data result](./Figure7.png)
<br>
Figure 7. Visualized results of the three real-world cryo-ET datasets. All of the images are selected from the center slices on x-, y- and z- axis, respectively.
<br>
![FSCe/o curve](./Figure8.png)
<br>
Figure 8. The FSC_e/o curves for the three real-world cryo-ET datasets. In each figure, the blue line points out the resolution of each method when r=0.5.

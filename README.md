#cuda_demosaicing
Demosaics RAW images on the GPU. This is the source code for the Cambridge MPhil project: https://www.dar.cam.ac.uk/drupal7/sites/default/files/Documents/publications/dcrr011.pdf.

Demosaicing is the process of converting raw sensor data into an RGB image. Sensors only collect only a single color at each pixel point, so the other RGB values have to be interpolated. However, this can lead to artefacts in the final image, in particular zippering on high contrast edges. More complex methods have emerged to prevent artefacts, the best being Adapative Homegenity Directed Demosaicing. But due to the complexity of the algorithm and the size of the images involved, demosaicing can be time consuming. This project is an investigation into how the GPU can be used to make this faster.

This project supports several demosaicing algoritihms including: Adapative Homegenity Directed Demosaicing, the algorithm with the highest quality results; Bilinear Interpolation, a fast and simple method; Edge Directed Demoasicing, a novel method of fast demosaicing that uses both AHD and Bilinear, using the more expensive method only on areas of the image where artefacts are likely to occur.

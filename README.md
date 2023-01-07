# SPEED-Pipeline
An implementation of Satellite Pre-processing Enhanced Edge Detection (SPEED) pipeline

For the full paper see [here](https://arxiv.org/pdf/2105.12633.pdf)

## Running the Code
Install requirements from the requirements.txt

Run the following: `python compareImageCannys.py --edge-detector hed_model --path ./`
Currently one `edge-detector` is given as a comparison point. It is not used within SPEED but rather acts as a comparison point
The `path` indicates where png images should be found, with the following structure
```
\Images
  \GroundTruth
    \image_name_gt.png
  \Original
    \image_name.png
```

Ground truth images can be supplied if you wish to see the SSIM against the SPEED pipeline. 
If not, just supply in some placeholder image in it's place (It must match in size)

## Paper Results
### Input images
![Input](https://github.com/Josh-Abraham/SPEED-Pipeline/blob/main/Results/Examples/Original_GT.png)
### Edge Detector Results
![Model Comparisons](https://github.com/Josh-Abraham/SPEED-Pipeline/blob/main/Results/Examples/Model_comparison.png)

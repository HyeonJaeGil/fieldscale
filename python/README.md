

## Installation

Install minimal dependencies using pip:
```
pip install numpy opencv-python
```

## Image Rescaling
Use the following code to rescale a thermal infrared image:

```python
import cv2
from fieldscale import Fieldscale

## Parameters used in the paper: 
## Please use the following parameters for the reproduction
# params = {
#     'max_diff': 100,
#     'min_diff': 100,
#     'iteration': 7,
#     'gamma': 1.5,
#     'clahe': True,
#     'video': False
# }

params = {
    'max_diff': 400,
    'min_diff': 400,
    'iteration': 7,
    'gamma': 1.5,
    'clahe': True,
    'video': False
}

input = './assets/demo.tiff' # path to the input image
fieldscale = Fieldscale(**params)
rescaled = fieldscale(input) # input can be either a path to an image or a uint16 numpy array
cv2.imshow('rescaled', rescaled)
cv2.waitKey(0)
```

## Video Rescaling
Please note that Fieldscale is designed to process a single image. But you can apply Fieldscale to a video by processing each frame sequentially. If you want to rescale a video, set `video` to `True` and provide the path to the video file:

```python
import os
import cv2
from fieldscale import Fieldscale

params_video = {
    'max_diff': 400,
    'min_diff': 400,
    'iteration': 7,
    'gamma': 1.5,
    'clahe': False,
    'video': True
}
folder_path = './assets/video' # path to the folder containing video frames
inputs = [os.path.join(folder_path, file) for file in sorted(os.listdir(folder_path))]
fieldscale = Fieldscale(**params_video)
for input in inputs:
    rescaled = fieldscale(input)
    cv2.imshow('rescaled', rescaled)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
```

### Parameter Tuning Guide
Fieldscale has several parameters that can be adjusted to suit your needs:
- `max_diff`, `min_diff`: $T_\text{LES}$ value in the paper. Depending on the input image, the optimal value may vary between 100 and 400. If you want to suppress the extreme parts and maximize the global consistency, set `max_diff` to a lower value (around 100). If you want to preserve the local details, set `max_diff` to a higher value (around 400).
- `iteration`: The number of iterations to apply the message passing. The optimal value is 7.
- `gamma`: The gamma value for gamma correction. The optimal value is 1.5.
- `clahe`: Whether to apply CLAHE. CLAHE can enhance the contrast of the image, but it may not be necessary for some images.
- `video`: Whether to process a video. If set to `True`, fields are smoothed to produce temporally consistent results.

For more details, please refer to the [paper](https://arxiv.org/abs/2405.15395) :)
# Face Swap: InsightFace + GFPGAN

Used:  
https://github.com/deepinsight/insightface  
https://github.com/TencentARC/GFPGAN  

Tested on **Python 3.12**  

## Fix for basicsr
`nano venv/lib/python3.12/site-packages/basicsr/data/degradations.py`  
replace line  
`from torchvision.transforms.functional_tensor import rgb_to_grayscale`  
to:  
`from torchvision.transforms.functional import rgb_to_grayscale`

## Usage
~~~
python face_swap.py \
--input "images/the-friends.jpg" \
--face_input "images/elon_musk.jpg" \
--output "output/out1.jpg"
~~~

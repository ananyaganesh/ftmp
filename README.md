# Future Talk Move Prediction
Code for future talk move prediction from the ACL 2021 Findings paper ["What Would a Teacher Do? Predicting Future Talk Moves"](https://arxiv.org/abs/2106.05249)

Model settings can be configured in `experiments.conf`, including specifying if pretrained layers need to be used. 

To use the GPU, change device from 'cpu' to 'cuda' in `train.py`, `models.py` and `nn_blocks.py`

Train models using `python train.py --expr name_of_experiment`, e.g. `python train.py --expr retrain`. Specify the location of where to save the model in `log_dir` of the experiment configuration.

Evaluate models using `python evaluation.py name_of_experiment`.


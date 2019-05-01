# French-to-English translation with Pytorch

*Author: Tianxiang Gao*

Pytorch implementation of seq2seq French-to-English translation. 

## Quick Start
Run French-to-English translation on GPU:0, the hyperparameters are defined in the `./experiments/attention/f2e/config.json'`
```
python main.py --gpu 0 --model_dir experiment/attention/f2e/
```

After the training, the best checkpoint will be stored in `./experiment/attention/f2e/ckpts/best.pth.tar`. To evaluate the
model on GPU:0, use the best checkpoint, beam search with width 3, and save attention weights heat map, simply run the 
following command:
```
python main.py --gpu 0 --model_dir experiment/attention/f2e/ --mode test --heatmap --beam_size 3
```

## How to use

### Train 
```
python main.py --gpu [gpu_id] --model_dir [model_dir]
```
In the `experiments/attention/` folder, there are two configuration json files for French-English and English-French 
translation. It is easy to set different hyper-parameters and play around the model.

Optional arguments for training:

- `--resume` whether to resume training from check point


### Test
```
python main.py --gpu [gpu_id] --model_dir [model_dir] --mode test
```

Optional arguments:

- `--beam_size` or `--bs` whether use beam search. Greedy search if the value is 0. Otherwise, the value is the width 
of beam search.
- `--heatmap` or `--hm` whether to generate and store the attention weight heat map.

## References
- [Pytorch official seq2seq tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-download-intermediate-seq2seq-translation-tutorial-py)
- [Stanford CS230 code examples](https://cs230-stanford.github.io/project-code-examples.html)

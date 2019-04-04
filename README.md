# French-to-English translation with Pytorch

*Author: Tianxiang Gao*

Pytorch implementation of seq2seq French-to-English translation. 
An adaptation from Pytorch official seq2seq [tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-download-intermediate-seq2seq-translation-tutorial-py).

## Quick Start
Run French-to-English translation on GPU:0, the hyperparameters are defined in the `./experiments/base_model/params.json'`
```
python train.py --gpu 0 --model_dir experiment/base_model
```

After the training, the best checkpoint will be stored in `./experiment/base_model/ckpts/best.pth.tar`. To evaluate the
model on GPU:0 and use the best checkpoint, simply run the command:
```
python evaluate.py --gpu 0 --model_dir experiment/base_model
```

## How to use

1. To train 
```
python train.py --gpu [gpu_id] --model_dir [model_dir]
```

2. To evaluate
```
python evaluate.py --gpu [gpu_id] --model_dir [model_dir]
```

## Credit
- [Pytorch official seq2seq tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-download-intermediate-seq2seq-translation-tutorial-py)
- [Stanford CS230 code examples](https://cs230-stanford.github.io/project-code-examples.html)

Introduction
---
The source code for paper **Delving Deeper into Clean Samples for Combating Noisy Labels**

How to use
---
The code is currently trained only on GPU and contains three different noise types of CIFAR dataset: `[CCN, OOD, IDN]`. You can specify the 
noise construction of noise dataset through `args.noise`.

- Demo
  - If you want to train our model in IDN CIFAR-10 dataset, you can modify `arg.noise`, `args.noise_type`, `args.noise_rate`, 
    `args.dataset`, and then run
      ```
      CUDA_VISIBLE_DEVICES=0 python train.py --noise idn  --noise_rate 0.2 \
        --forget_rate 0.2  --noise_type symmetric --dataset cifar10  \
        --train_nums 3 --lr 0.01 --n_epoch 100  --epoch_decay_start 50 \
        --num_gradual 10  --up_gradual 20
      ```
      

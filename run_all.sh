#!/bin/bash

datasets=("wikipedia" "uci" "reddit")
# datasets=("enron" "mooc" "Contacts")

for dataset in "${datasets[@]}" 
do
    echo "Running on dataset: $dataset"

    # # real data
    for sample in {1..10}; do
        python main.py  -d "$dataset" --pos_dim 108 --bs 32 --n_degree 64 1 --mode t --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --gpu 0
        python main.py  -d "$dataset" --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --gpu 0
    done 

    for sample in {1..10}; do
        distort="shuffle_${sample}_"
        python main.py -d "$dataset" --pos_dim 108 --bs 32 --n_degree 64 1 --mode t --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --gpu 0 --distortion "$distort"
        python main.py -d "$dataset" --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --gpu 0 --distortion "$distort"
    done

    # # distorted data: all samples
    for sample in {1..10}; do
        distort="intense_5_${sample}_"
        python main.py -d "$dataset" --pos_dim 108 --bs 32 --n_degree 64 1 --mode t --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --gpu 0 --distortion "$distort"
        python main.py -d "$dataset" --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --gpu 0 --distortion "$distort"
    done

done


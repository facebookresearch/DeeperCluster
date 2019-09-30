# DeeperCluster: Unsupervised Pre-training of Image Features on Non-Curated Data 

This code implements the unsupervised pre-training of convolutional neural networks, or convnets, as described in [Unsupervised Pre-training of Image Features on Non-Curated Data](https://arxiv.org/abs/1905.01278).

## Models
We provide for download the following models:
* DeeperCluster model trained on the full YFCC100M dataset;
* DeepCluster [2] model trained on 1.3M images subset of the YFCC100M dataset;
* RotNet [3] model trained on the full YFCC100M dataset;
* RotNet [3] model trained on ImageNet dataset without labels.

All these models follow a standard VGG-16 architecture with batch-normalization layers.
Note that in Deep/DeeperCluster models, sobel filters are computed within the models as two convolutional layers (greyscale + sobel filters).
The models expect RGB inputs that range in [0, 1]. You should preprocess your data before passing them to the released models by normalizing them: ```mean_rgb = [0.485, 0.456, 0.406]```; ```std_rgb = [0.229, 0.224, 0.225] ```.

| Method / Dataset   |      YFCC100M      |  ImageNet |
|--------------------|--------------------|-----------|
| DeeperCluster | [ours](https://dl.fbaipublicfiles.com/deepcluster/ours/ours.pth) | - |
| DeepCluster | [deepcluster_yfcc100M](https://dl.fbaipublicfiles.com/deepcluster/deepcluster/deepcluster_flickr.pth) trained on 1.3M images | [deepcluster_imagenet](https://dl.fbaipublicfiles.com/deepcluster/vgg16/checkpoint.pth.tar) (found [here](https://github.com/facebookresearch/deepcluster)) |
| RotNet | [rotnet_yfcc100M](https://dl.fbaipublicfiles.com/deepcluster/rotnet/rotnet_flickr.pth) | [rotnet_imagenet](https://dl.fbaipublicfiles.com/deepcluster/rotnet/rotnet_imagenet.pth) |

To automatically download all models you can run:
```
$ ./download_models.sh
```

## Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install 1.0.0
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension
- [Faiss](https://github.com/facebookresearch/faiss) GPU install
- Download [YFCC100M dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67&guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAI-kwr4-KyuBJKrOUt3nzqR8H9hxu4cel43rHsFuk_4mKhjPoepAekZ7thVhdnOX-oLYek43-YMLIGQ5xmyPzU0Rc--RJsuRMSvqzpxxpug7Mg7XEv15bBS030Ood5TfcXwna_hjdbCtiPeoCOl5Knhog71KhdWnrFwuX2TloFFJ). The ids of the 95.920.149 images we managed to download can be found [here](https://dl.fbaipublicfiles.com/deepcluster/flickr_unique_ids.npy). `wget -c -P ./src/data/ "https://dl.fbaipublicfiles.com/deepcluster/flickr_unique_ids.npy"`

## Unsupervised Learning of Visual Features

The script ```main.sh``` will run our method. Here is a screenshot:
```
python main.py

## handling experiment parameters
--dump_path ./exp/                  # Where to store the experiment

## network params
--pretrained PRETRAINED             # Use this instead of random weights

## data params
--data_path DATA_PATH               # Where to find YFCC100M dataset
--size_dataset 100000000            # How many images to use for training
--workers 10                        # Number of data loading workers
--sobel true                        # Apply Sobel filter

## optim params
--lr 0.1                            # Learning rate
--wd 0.00001                        # Weight decay
--nepochs 100                       # Number of epochs to run
--batch_size 48                     # Batch size per process

## model params
--reassignment 3                    # Reassign clusters every this epoch
--dim_pca 4096                      # Dimension of the pca on the descriptors
--super_classes 16                  # Total number of super-classes
--rotnet true                       # Network needs to classify large rotations

## k-means params
--k 320000                          # Total number of clusters
--warm_restart false                # Use previous centroids as init
--use_faiss true                    # Use faiss for E step in k-means
--niter 10                          # Number of k-means iterations

## distributed training params
--world-size 64                     # Number of distributed processes
--dist-url DIST_URL                 # Url used to set up distributed training
```

You can look the training full documentation up with ```python main.py --help```.

### Distributed training
This implementation, as it is, supports only distributed mode activated.
It has been specifically designed for multi-GPU and multi-node training and tested up to 128 GPUs distributed accross 16 nodes of 8 GPUs each.
You can run code in two different scenarios:

* 1- Submit your job to a computer cluster. This code is adapted for SLURM job scheduler but you can modify it for your own scheduler.

* 2- Put export `NGPU=xx; python -m torch.distributed.launch --nproc_per_node=$NGPU` before the python file you want to execute (with xx the number of gpus you want).
For example, to run an experiment with a single GPU on a single machine, simply replace `python main.py` with:
```
export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU main.py
```


The parameter `rank` is set automatically in both scenario in [utils.py](./src/utils.py#L42).

The parameter `local_rank` is more or less useless.

The parameter `world-size` needs to be set manually in scenario 1 and is set automatically in scenario 2.

The parameter `dist-url` needs to be set manually in both scenario. Refer to pytorch distributed [doc](https://pytorch.org/docs/stable/distributed.html) to set correctly the initialization method.


The total number of GPUs used for an experiment (```world-size```) must be divisible by the total number of super-classes (```super_classes```).
Hence, exactly a total of ```super_classes``` training communication groups of ```world_size / super_classes``` GPUs each are created.
The parameters of a sub-class classifier specific to a super-class are shared within the corresponding training group.
Each training group deals only with the subset of images and the rotation angle associated with its corresponding super-class.
For this reason, computing batch statistics in the batch normalization layers for *the entire batch* (distributed accross the different training groups) is crucial.
We do so thanks to [apex](https://github.com/NVIDIA/apex/tree/master/apex/parallel#synchronized-batch-normalization).

For the first stage of hierarchical clustering into ```nmb_super_clusters``` clusters, the entire pool of GPUs is used.
Then for the second stage, we create ```nmb_super_clusters``` clustering communication groups of ```world_size / nmb_super_clusters``` GPUs each.
Each of these clustering groups independantly performs the second stage of hierarchical clustering on its corresponding subset of data (data belonging to the associated super-cluster).

For example, as illustrated below, let's assume we want to run a training with 8 super-classes and we have access to a pool of 16 GPUs.
As many training distributed communication groups as the number of super-classes are created.
This corresponds to creating 8 training groups (in red) of 2 GPUs.
Moreover, the first level of the hierarchical k-means corresponds to the clustering of the data into 8/4=2 super-clusters.
Hence, 2 clustering groups (in blue) are created.
![distributed](./distributed_training.png)

You can have a look [here](./src/utils.py#L42) for more details about how we define the different communication groups.
The multi-node is automatically handled by SLURM.


### Running DeepCluster or RotNet
Our implementation is generic enough to encompass both DeepCluster and RotNet trainings.
* DeepCluster: set ```super_classes``` to ```1``` and ```rotnet``` to ```false```.
* RotNet: set ```super_classes``` to ```4```, ```k``` to ```1``` and ```rotnet``` to ```true```.

## Evaluation protocols

### Pascal VOC

To reproduce our results on PASCAL VOC 2007 classification task run:
* FC6-8
```
python eval_voc_classif.py --data_path $PASCAL_DATASET --fc6_8 true --pretrained downloaded_models/deepercluster/ours.pth --sobel true --lr 0.003 --wd 0.00001 --nit 150000 --stepsize 20000 --split trainval
```

* ALL
```
python eval_voc_classif.py --data_path $PASCAL_DATASET --fc6_8 false --pretrained downloaded_models/deepercluster/ours.pth --sobel true --lr 0.003 --wd 0.0001 --nit 150000 --stepsize 10000 --split trainval
```

**Running the experiment with 5 seeds.**
There are different sources of randomness in the code: classifier initialization, ramdon crops for the evaluation and training with CUDA.
For more reliable results, we recommend to run the experiment several times with different seeds (`--seed 36` for example).

**Hyper-parameters selection.**
We select the value of the different hyper-parameters (weight-decay `wd`, learning rate `lr`, and step-size `stepsize`) by training on the train split and validating on the validation set.
To do so, simply use `--split train`.

### Linear classifiers

We train linear classifiers with a logistic loss on top of frozen convolutional layers at different depths.
To reduce the influence of feature dimension in the comparison, we average-pool the features until their dimension is below 10k.

To reproduce our results from Table-3 run: `./conv13.sh`.

To reproduce our results from Figure-2 run: `./linear_classif_layers.sh`

**Learning rates.**
We use the learning rate decay recommended for linear models with L2 regularization by Leon Bottou in [Stochastic Gradient Descent Tricks](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf).

**Hyper-parameters selection.**
For experiments on Pascal, we select the value of the initial learning rate by training on the train split and validating on the validation set.
To do so, simply use `--split train`.
For experiments on ImageNet and Places, this code implements k-fold cross-validation.
Simply set `--kfold 3` for 3-fold cross-validation.
Then set `--cross_valid 0` for training on splits 1 and 2 and validating on split 0 for example.

**Checkpointing and distributed training.**
This code implements automatic checkpointing and is adapted to distributed training on multi-gpus and/or multi-nodes.

### Pre-training for ImageNet

To reproduce our results on the pre-training for ImageNet experiment (Table-2) run:

```
mkdir -p ./exp/pretraining_imagenet/
export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU eval_pretrain.py --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --sobel2RGB true --nepochs 100 --batch_size 256 --lr 0.1 --wd 0.0001 --dump_path ./exp/pretraining_imagenet/ --data_path $DATAPATH_IMAGENET
```

**Checkpointing and distributed training.**
This code implements automatic checkpointing and is specifically intended for distributed training on multi-gpus and/or multi-nodes.
The results in the paper for this experiment are obtained with training on 4 GPUs (the batch size per GPU is 64 in this case).

## References

### Unsupervised Pre-training of Image Features on Non-Curated Data

[1] M. Caron, P. Bojanowski, J. Mairal, A. Joulin [Unsupervised Pre-training of Image Features on Non-Curated Data](https://arxiv.org/abs/1905.01278)
```
@inproceedings{caron2019unsupervised,
  title={Unsupervised Pre-Training of Image Features on Non-Curated Data},
  author={Caron, Mathilde and Bojanowski, Piotr and Mairal, Julien and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2019}
}
```


### Deep clustering for unsupervised pre-training of visual features

[code](https://github.com/facebookresearch/deepcluster)

[2] M. Caron, P. Bojanowski, A. Joulin, M. Douze [*Deep clustering for unsupervised learning of visual features*](http://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html)
```
@inproceedings{caron2018deep,
  title={Deep clustering for unsupervised learning of visual features},
  author={Caron, Mathilde and Bojanowski, Piotr and Joulin, Armand and Douze, Matthijs},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

### Unsupervised representation learning by predicting image rotations
[code](https://github.com/gidariss/FeatureLearningRotNet)

[3] S. Gidaris, P. Singh, N. Komodakis [*Unsupervised representation learning by predicting image rotations*](https://openreview.net/forum?id=S1v4N2l0-)

```
@inproceedings{
  gidaris2018unsupervised,
  title={Unsupervised Representation Learning by Predicting Image Rotations},
  author={Spyros Gidaris and Praveer Singh and Nikos Komodakis},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=S1v4N2l0-},
}
```

## License

See the [LICENSE](LICENSE) file for more details.

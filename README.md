<p align="middle">
  <img src="assets/scaling-law.png" width="300"/>
  <img src="assets/architecture.png" width="300"/>
</p>


# Wukong for large-scale recommendation

Implements the paper [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545v1) from Meta.

It presents a novel state-of-the-art architecture for recommendation systems that additionally follows a similar scaling law of large language models, where the model performance seems to increase with respect to the model scale without a clear asymptote on the scales explored in the paper.

Only a pytorch implementation is presented here for now but eventually a Tensorflow implementation will be added.

## Usage <a name = "usage"></a>

```python
from pytorch import Wukong


# mock input data
BATCH_SIZE = 1024
NUM_EMBEDDING = 10_000 # vocab size
NUM_CAT_FEATURES = 32
NUM_DENSE_FEATURES = 16

sparse_inputs = torch.multinomial(
    torch.rand((BATCH_SIZE, NUM_EMBEDDING)),
    NUM_CAT_FEATURES,
    replacement=True,
)
dense_inputs = torch.rand(BATCH_SIZE, NUM_DENSE_FEATURES)

# takes hyperparameters from the paper
model = Wukong(
    num_layers=3,
    num_emb=NUM_EMBEDDING,
    dim_emb=128,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=16,
    num_emb_fmb=16,
    rank_fmb=24,
    num_hidden_wukong=2,
    dim_hidden_wukong=512,
    num_hidden_head=2,
    dim_hidden_head=512,
    dim_output=1
)

# outputs are the logits and will need to be rescaled with a sigmoid to get a probability
outputs = model(sparse_inputs, dense_inputs)

```

## Citations

```bibtex
@misc{zhang2024wukong,
      title={Wukong: Towards a Scaling Law for Large-Scale Recommendation}, 
      author={Buyun Zhang and Liang Luo and Yuxin Chen and Jade Nie and Xi Liu and Daifeng Guo and Yanli Zhao and Shen Li and Yuchen Hao and Yantao Yao and Guna Lakshminarayanan and Ellie Dingqiao Wen and Jongsoo Park and Maxim Naumov and Wenlin Chen},
      year={2024},
      eprint={2403.02545},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

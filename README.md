<p align="middle">
  <img src="assets/scaling-law.png" width="300"/>
  <img src="assets/architecture.png" width="300"/>
</p>


# Wukong for large-scale recommendation

Implements the paper [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545v1) from Meta.

It presents a novel state-of-the-art architecture for recommendation systems that additionally follows a similar scaling law to large language models, where the model performance seems to increase with respect to the model scale without a clear asymptote on the scales explored in the paper.

Only a pytorch implementation is presented here for now but eventually a Tensorflow implementation will be added.

note: for simplicity sake, only categorical inputs are supported in this implementation. To handle both categorical and continuous inputs, one will need to projects dense inputs to the same embedding space as the sparse ones and concatenate the two together to obtain the "Dense Embeddings" of the paper.

## Usage <a name = "usage"></a>

```python
from pytorch import Wukong


# mock input data
BATCH_SIZE = 1024
NUM_EMBEDDING = 10_000 # vocab size
NUM_FEATURES = 32

inputs = torch.multinomial(
    torch.rand((BATCH_SIZE, NUM_EMBEDDING)),
    NUM_FEATURES,
    replacement=True,
)

# takes hyperparameters from the paper
model = Wukong(
    num_layers=3,
    num_emb=NUM_EMBEDDING,
    dim_emb=128,
    dim_input=NUM_FEATURES,
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
outputs = model(inputs)

```

# Citations

- [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545v1)

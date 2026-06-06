# ConnectX: PPO vs GRPO (self-play)

Two policy-gradient methods trained by self-play on Kaggle's ConnectX
(Connect-4), implemented from scratch in PyTorch and compared head-to-head.

Both methods share everything except how the advantage is estimated:

| component            | shared design                                              |
|----------------------|------------------------------------------------------------|
| state representation | 3-channel canonical board `(own / opponent / empty)`       |
| action masking       | illegal (full) columns masked to `-inf` in the policy      |
| network backbone     | residual conv trunk (64ch) + FC                            |
| self-play            | opponent sampled from snapshots + random + Kaggle negamax  |
| objective            | PPO-style clipped surrogate + entropy bonus                |

| method | advantage estimate                                        | critic |
|--------|-----------------------------------------------------------|--------|
| **PPO**  | GAE(λ) from a learned value baseline                    | yes    |
| **GRPO** | group-relative normalized returns `(R-mean)/(std+eps)` | no     |

## Layout

```
course/
├── rl/
│   ├── env.py        # Connect-4 engine, canonical encoding, SelfPlayEnv, Policy
│   ├── networks.py   # ConvBackbone, ActorCritic (PPO), PolicyNetwork (GRPO)
│   ├── policies.py   # NetPolicy: wrap a net as a Policy (eval / pool opponent)
│   ├── pool.py       # self-play opponent snapshot pool
│   ├── negamax.py    # alpha-beta opponent + Kaggle negamax clone
│   ├── ppo.py        # PPOAgent  (actor-critic + GAE)
│   ├── grpo.py       # GRPOAgent (critic-free, group-relative advantage)
│   ├── evaluate.py   # internal + Kaggle-faithful evaluation
│   └── configs.py    # build_agent("ppo" | "grpo")
├── train.py          # train an agent by self-play, save curve + checkpoint
├── compare.py        # curves + head-to-head + final win rates (+ --kaggle)
└── demo.ipynb        # original Kaggle getting-started notebook
```

## Usage

```bash
# train PPO and GRPO, then generate plots and Kaggle comparisons
bash run.sh

# override defaults when needed
STEPS=500000 EVAL_GAMES=200 bash run.sh

# or one at a time
CUDA_VISIBLE_DEVICES=6 python train.py --exp ppo  --steps 300000
CUDA_VISIBLE_DEVICES=7 python train.py --exp grpo --steps 300000

# compare: plot curves, head-to-head, final win rates -> results/compare.png
CUDA_VISIBLE_DEVICES=6 python compare.py --kaggle
```

Reward convention: +1 win / −1 loss / 0 draw, from the learner's perspective.
Training metric curves log win rate vs `random` and vs a depth-4 Kaggle-style
`negamax` benchmark every `--eval-every` self-play steps.

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
├── connectx/             # the library (importable package)
│   ├── env.py            # Connect-4 engine, canonical encoding, SelfPlayEnv, Policy
│   ├── networks.py       # ConvBackbone, ActorCritic (PPO), PolicyNetwork (GRPO), NetPolicy
│   ├── negamax.py        # faithful Kaggle negamax clone (opponent + benchmark)
│   ├── pool.py           # self-play opponent snapshot pool + curriculum weights
│   ├── ppo.py            # PPOAgent  (actor-critic + GAE)
│   ├── grpo.py           # GRPOAgent (critic-free, group-relative advantage)
│   ├── evaluate.py       # internal + Kaggle-faithful evaluation
│   └── __init__.py       # build_agent("ppo" | "grpo") factory + exports
├── scripts/
│   ├── train.py          # train an agent by self-play, save curve + checkpoint
│   ├── compare.py        # curves + head-to-head + final win rates (+ --kaggle)
│   └── make_submission.py# embed a checkpoint into a self-contained Kaggle agent
├── submissions/
│   ├── ppo.py · grpo.py  # self-contained Kaggle agents (weights embedded)
│   ├── validate.py       # run a submission the way Kaggle does, before submitting
│   └── kaggle_submit.ipynb
├── checkpoints/          # trained agents (class-independent .pt: arch + state_dict)
├── results/              # training curves + plots
└── train.sh              # one-command pipeline (train both -> compare -> plots)
```

## Usage

```bash
# train PPO and GRPO, then generate plots and Kaggle comparisons
bash train.sh

# override defaults when needed
STEPS=500000 EVAL_GAMES=200 bash train.sh

# or one at a time
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --exp ppo  --steps 300000
CUDA_VISIBLE_DEVICES=7 python scripts/train.py --exp grpo --steps 300000

# compare: plot curves, head-to-head, final win rates -> results/compare.png
CUDA_VISIBLE_DEVICES=6 python scripts/compare.py --kaggle
```

## Kaggle submission

`submissions/ppo.py` and `submissions/grpo.py` are self-contained agents with the
trained weights embedded — a pure forward pass, no search. Submit the stronger
one (PPO):

```bash
python submissions/validate.py                  # sanity-check before submitting
kaggle competitions submit -c connectx -f submissions/ppo.py -m "PPO self-play"
# or upload submissions/kaggle_submit.ipynb, Run All, then "Submit to Competition"

# regenerate from a checkpoint after retraining:
python scripts/make_submission.py --ckpt checkpoints/ppo_s0.pt --out submissions/ppo.py --method PPO
```

Reward convention: +1 win / −1 loss / 0 draw, from the learner's perspective.
Training metric curves log win rate vs `random` and vs a depth-4 Kaggle-style
`negamax` benchmark every `--eval-every` self-play steps.

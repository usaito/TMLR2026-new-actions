## Offline Contextual Bandits in the Presence of New Actions

This repository contains the code to replicate experiments conducted in the paper "[Offline Contextual Bandits in the Presence of New Actions](https://openreview.net/forum?id=3YTlnocq4P)".

- **OpenReview:** https://openreview.net/forum?id=3YTlnocq4P
- **arXiv:** https://arxiv.org/abs/2605.18509
- **Venue:** Transactions on Machine Learning Research (TMLR) — Accepted

## Abstract
Automated decision-making algorithms drive applications in domains such as recommendation systems and search engines. These algorithms often rely on off-policy contextual bandits or *off-policy learning* (OPL). Conventionally, OPL selects actions that maximize the expected reward from an existing action set. However, in many real-world scenarios, actions—such as news articles or video content—change continuously, and the action space evolves over time compared to when the logged data was collected. We define actions introduced after deploying the logging policy as *new actions* and focus on the problem of OPL with new actions. Existing OPL methods identify optimal actions from the existing set effectively. However, these methods struggle to learn and select new actions because no logged data exists for these actions. To address this limitation, we propose a new OPL method that leverages action features. In particular, we introduce the Local Combination PseudoInverse (LCPI) estimator for the policy gradient, generalizing the PseudoInverse estimator initially proposed for off-policy evaluation of slate bandits. LCPI controls the trade-off between reward-modeling condition and the condition for data collection regarding the action features, capturing the interaction effects among different dimensions of action features. Furthermore, we propose a generalized algorithm called **Policy Optimization for Effective New Actions (PONA)**, which integrates LCPI, a component specialized for new action selection, with Doubly Robust (DR), which excels at learning within existing actions. We define PONA as a weighted sum of the LCPI and DR estimators, optimizing both the selection of existing and new actions, and allowing the proportion of new action selections to be adjusted by controlling the weight parameter. Through extensive experiments, we demonstrate that PONA efficiently selects new actions while maintaining the overall policy performance as opposed to most existing methods that cannot select new actions.

## Citation

If you use this code or build on this work, please cite the paper as follows:

```bibtex
@article{kishimoto2026offlinecontextualbandits,
  title   = {Offline Contextual Bandits in the Presence of New Actions},
  author  = {Ren Kishimoto and Tatsuhiro Shimizu and Kazuki Kawamura and Takanori Muroi and Yusuke Narita and Yuki Sasamoto and Kei Tateno and Takuma Udagawa and Yuta Saito},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=3YTlnocq4P}
}
```

## Dependencies
This repository supports Python 3.10.6 or newer.

- numpy==1.23.5
- pandas==1.5.2
- scikit-learn==1.1.3
- matplotlib==3.7.1
- torch==1.12.0


## Running the Code
The commands needed to reproduce the experiments are summarized below. Please move under the `synthetic` directly first and then run the notebooks.

### Synthetic Data

```bash
# How does PONA perform with varying training data sizes?
synthetic/main/main_num_trains.ipynb
synthetic/show/show_num_trains.ipynb

# How does PONA perform with varying percentages of new actions?
synthetic/main/main_num_actions.ipynb
synthetic/show/show_num_actions.ipynb

# How does PONA perform with varying degrees of local linearity violation?
synthetic/main/main_gamma.ipynb
synthetic/show/show_gamma.ipynb

# How does PONA perform with varying lower limits on the proportion of new actions?
synthetic/main/main_constrain.ipynb
synthetic/show/show_rho_L.ipynb

# How does PONA perform with varying upper limits on the proportion of new actions?
synthetic/main/main_constrain.ipynb
synthetic/show/show_rho_U.ipynb

```

### Real-World Data

We use [KuaiRec dataset](https://kuairec.com/). Please download the above datasets from the repository and put them under `./kuairec/data/` as follows.

```
src/
├ synthetic/
├ kuairec/
    └ data/
        ├ small_matrix.csv
        ├ item_categories.csv
        ├ kuairec_caption_category.csv
        ├ user_feature.csv
```

Then, run the following notebooks.

```bash
kuairec/processing_data.ipynb

# How does PONA perform with varying training data sizes?
kuairec/main/main_num_trains.ipynb
kuairec/show/show_num_trains.ipynb

# How does PONA perform with varying percentages of new actions?
kuairec/main/main_num_actions.ipynb
kuairec/show/show_num_actions.ipynb

# How does PONA perform with varying varying lower limits on the proportion of new actions?
kuairec/main/main_constrain.ipynb
kuairec/show/show_rho_L.ipynb

# How does PONA perform with varying varying upper limits on the proportion of new actions?
kuairec/main/main_constrain.ipynb
kuairec/show/show_rho_U.ipynb

```
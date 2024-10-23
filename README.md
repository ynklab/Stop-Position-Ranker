# Stop-Position-Ranker

## Overview
This project aims to create a novel architecture and pipeline leveraging existing models to predict stop positions using [scene graphs](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Unbiased_Scene_Graph_Generation_From_Biased_Training_CVPR_2020_paper.pdf). Our approach focuses on enhancing decision-making around stop positions by introducing the ability to rank multiple candidates and provide justifications for not selecting certain positions.

## Data Collection
To address the limitations of the current [dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), we are planning to augment the data with:

- Multiple Stop Position Candidates: Collect data that includes various stop position options for each scene.
- Ranking Annotations: Add annotations that rank the candidates based on appropriateness for stopping.
- Non-Selection Reasons: Annotate why certain stop positions are not suitable (e.g., obstructed view, hazardous conditions).

This enhanced dataset will enable our model to learn a more nuanced decision-making process, improving stop position prediction performance.

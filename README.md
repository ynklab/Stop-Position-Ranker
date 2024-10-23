# Vision-and-Language-Navigation

## Overview
This project aims to create a novel architecture and pipeline leveraging existing models to predict stop positions using scene graphs. Our approach focuses on enhancing decision-making around stop positions by introducing the ability to rank multiple candidates and provide justifications for not selecting certain positions.

## Data Collection
To address the limitations of the current [dataset](dataset), we are planning to augment the data with:

- Multiple Stop Position Candidates: Collect data that includes various stop position options for each scene.
- Ranking Annotations: Add annotations that rank the candidates based on appropriateness for stopping.
- Non-Selection Reasons: Annotate why certain stop positions are not suitable (e.g., obstructed view, hazardous conditions).

This enhanced dataset will enable our model to learn a more nuanced decision-making process, improving stop position prediction performance.

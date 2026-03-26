# Multi-Objective-Optimization-of-Steel

[**中文版 README**](readme_zh.md)  |  [**日本語版 README**](readme_jp.md) 

## Overview

This repository provides the official implementation of the paper:

“Multi-Objective Optimization of Corrosion Resistance, Strength, Ductility Properties of Weathering Steel Utilizing Interpretable Attention-Based Deep Learning Model” (published in npj Materials Degradation, https://doi.org/10.1038/s41529-025-00654-y)

The repository includes:

- `feature_select.py` for extracting features from xlsx file, using attention model
- `feature_sort_attention.py` for sorting features
- `mlp_predition_all.py` for predicting the features that satisfy the first author's requirements

This first author selected two components to be used in steelmaking and conducted a series of experiments.

## Clarification Statement

I previously served as a technical contributor to this project, independently writing all the code and executing the experimental results using the servers at my affiliated laboratory. Additionally, I assisted with the response/rebuttal process during the peer review of this paper.

To clarify the facts and uphold the principles of academic rigor, I am issuing the following statement:

### Disclaimer of Responsibility and Institutional Separation
I have proactively removed this paper from my Curriculum Vitae and Google Scholar profile. I have not used, nor will I use, this paper for any personal academic benefit. Any errors, inaccuracies, or lack of academic rigor in the final published version are entirely unrelated to me, as well as to my affiliated institutions: Southeast University and Institute of Science Tokyo.

### Unauthorized Modification of Figures
"Figure 12" in this work was originally created by me. However, the first author altered the figure's style for the final publication without my knowledge or confirmation (specifically, the connection line between 'norm' and 'V'). For verification, I have uploaded screenshots of my original version, along with the modification log files from my local machine, to this repository.

### Critical Discrepancies Between Implementation and Claims
As the sole developer of the code, I must highlight fundamental discrepancies between the claims made in the paper and the actual technical implementation. The first author, who was not substantially involved in the technical execution, failed to disclose these critical issues:

- Attention Mechanism Discrepancy: The paper claims to utilize Multi-head Attention. However, a review of the code in this repository clearly confirms that only Single-head Attention was actually implemented.

- Data Volume and Model Overfitting: The dataset size used in this study is objectively insufficient to train an effective hierarchical/multi-head attention model from scratch. The open-sourced model provided here is, in fact, severely overfitted—a fatal technical limitation that was entirely omitted from the paper.

By providing this codebase and the associated logs, my sole intention is to present the factual technical reality of this work. Readers are encouraged to compare the open-source code with the manuscript to independently assess the scientific credibility of the paper.

## P.S.

Out of trust built over a three-year relationship, I unreservedly took on all the coding and plotting for this work. It is deeply chilling that the first author's only demands were "I just want a picture" or "I just want it to be more complex," constantly rushing the delivery without any understanding of—or interest in—how the models actually worked. This superficial approach to academia is the fundamental reason I am stepping forward today.

Regarding the severe breaches of trust in our personal relationship (including multiple instances of infidelity and a seamless transition to a new partner), the subsequent gaslighting accusing me of "emotional neglect," and the so-called legal warnings from her family—I have no intention of providing further moral commentary here. To make a clean break, I have encrypted and archived the relevant non-academic evidence and placed it in the repository's Release section, solely to purge these files from my personal devices. I wish you peace of mind in your new life, if that is even possible.

# Hate Speech Detection Comparative Study: GPT-3.5 vs. Fine-tuned BERT Model

This repository contains the code for our project that compares the performance of hate speech detection using two state-of-the-art language models: GPT-3.5 and a fine-tuned BERT model. Our investigation leverages the robust features of each model to understand their strengths and limitations in identifying toxic language patterns.

Within the `hate_speech_detection_pipeline` folder, you will find all the necessary notebooks and scripts to preprocess data, conduct exploratory data analysis (EDA), run the classification models, and evaluate their performance. The folder `workbench` contains supplementary tools for corpus analysis and text preprocessing but they are unrelated to our research poster.

The fine-tuning and evaluation are based on the Jigsaw Toxic Comment Classification Challenge dataset and the distilled version of BERT (DistilBERT), aimed at being smaller, faster, and more resource-efficient while retaining BERT's capabilities.

You can look into our methodology and findings by exploring the included Jupyter notebooks and Python scripts. The results are stored in the `results` directory.

## Project Structure

```plaintext
.
└── hate_speech_detection_pipeline
   ├── apiGPT.py
   ├── classification_evaluation.ipynb
   ├── data_preprocessing.ipynb
   ├── distilBERT.py
   ├── BERT.py
   ├── eda.ipynb
   ├── fine_tune_distilBERT.ipynb
   ├── classification_evaluation_pipeline.ipynb
   └── results
       └── data_labeled_distilBERT_GPTzeroshot_GPT_viewshot_BERT.csv

```
## Citations



1) cjadams, J. Sorensen, J. Elliott, L. Dixon, M. McDonald, nithum, W. Cukierski, "Toxic Comment Classification Challenge," Kaggle, 2017. Available: https://kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge.

2) V. Sanh, L. Debut, J. Chaumond, T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," 2020. [Online]. Available: arXiv:1910.01108.


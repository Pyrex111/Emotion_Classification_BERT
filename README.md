# Emotion Classification with BERT

**CSC 503 Data Mining team project | University of Victoria | Winter 2025**

This repository contains the BERT component of a group project on emotion classification using the EmpatheticDialogues dataset. It fine-tunes `bert-base-uncased` to classify conversations into 19 emotion categories after grouping related labels. The repository includes the training notebook, dataset files, group progress and final reports, and final presentation.

The broader team project compared traditional classifiers, a hierarchical classification model, and BERT. The runnable code included here is the BERT notebook; the reports document the broader experiments and class-correlation analysis.

## Team and contributions

| Team member | Contribution |
| --- | --- |
| **Taimoor Shahzad** | **Class-correlation analysis and BERT-based classification**, including BERT model development, fine-tuning, and evaluation. Also contributed TF-IDF baseline experiments with balanced-class Logistic Regression and Random Forest, SVM-RBF with SMOTE, and LightGBM with Bayesian optimization. |
| **Nitin Gupta** | Data preprocessing and a TF-IDF Logistic Regression baseline. |
| **Akina Kuperus** | Word2Vec and Doc2Vec experiments with Logistic Regression and linear SVM, including hyperparameter tuning and evaluation. |
| **Riya Shrestha** | Baseline classification, tuning, and error/confusion-matrix reporting; Random Forest and Naive Bayes experiments. |
| **Abdul-Wahid Yakubu** | Decision Tree baseline using the Gini criterion. |

The [progress report](docs/csc503_group_project_progress.pdf), sections 4 and 5, records the mid-project task breakdown and initial experiments. The [final report](11April-CSC_503_Final_Project.pdf) describes the later class-correlation analysis, emotion-label grouping, hierarchical model, and BERT results. Taimoor's main contributions were class-correlation analysis and the BERT classification component.

## Class correlation and emotion grouping

The final report examines confusion between similar emotion classes and visualizes their relationships with a correlation heatmap. This analysis informed the reduction from 32 original labels to 19 grouped classes, including pairs or groups such as `afraid`/`terrified` and `angry`/`annoyed`/`furious`. The BERT notebook applies the emotion mapping to training, validation, and test labels before classification.

## Approach

The notebook groups utterances by conversation, combines each prompt with its conversation text, merges selected emotion labels into broader categories, and encodes the resulting labels. It fine-tunes BERT using the `[CLS]` representation, a dropout layer, and a linear classification head.

Training uses cross-entropy loss, AdamW, and a cosine annealing learning-rate schedule. The checkpoint with the lowest validation loss is saved and then evaluated on the test set. The notebook produces a classification report, a confusion matrix, training and validation error curves, and examples of misclassified conversations.

| Setting | Notebook value |
| --- | --- |
| Pretrained model | `bert-base-uncased` |
| Maximum token length | 128 |
| Batch size | 16 |
| Epochs | 5 |
| Initial learning rate | `2e-5` |
| Weight decay | `0.01` |
| Dropout | `0.5` |

## Repository contents

| Path | Contents |
| --- | --- |
| `BERT_FINAL_28MARCH.ipynb` | Data preparation, BERT fine-tuning, evaluation, and visualizations. |
| `empatheticdialoguesdataset/train.csv` | Supplied training data. |
| `empatheticdialoguesdataset/valid.csv` | Supplied validation data. |
| `empatheticdialoguesdataset/test.csv` | Supplied test data. |
| `docs/csc503_group_project_progress.pdf` | Original mid-project progress report, task breakdown, and initial results. |
| `11April-CSC_503_Final_Project.pdf` | Final project report. |
| `CSC 503 Final Presentation.pdf` | Final presentation in PDF format. |
| `CSC 503 Final Presentation.pptx` | Editable final presentation. |

## Run in Google Colab

1. Open [`BERT_FINAL_28MARCH.ipynb`](BERT_FINAL_28MARCH.ipynb) in Google Colab and select a GPU runtime. The original notebook was configured for a T4 GPU.
2. Upload the three CSV files to a folder named `DatasetFiles` in your Google Drive, or edit `train_file`, `valid_file`, and `test_file` in the notebook to match their location.
3. Install the imported dependencies in a Colab cell:

   ```python
   %pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision torchaudio "transformers[torch]" tensorflow
   ```

4. Run the notebook cells in order and authorize the Google Drive mount when prompted. The tokenizer and pretrained BERT weights download from Hugging Face on the first run.

The original paths are:

```text
/content/drive/MyDrive/DatasetFiles/train.csv
/content/drive/MyDrive/DatasetFiles/valid.csv
/content/drive/MyDrive/DatasetFiles/test.csv
```

Training writes `bert_emotion_classifier_best.pth` to the runtime. Evaluation writes `classification_report_test_set.csv`, `Test Set Confusion Matrix.jpg`, and `Training & Validation Errors vs Epochs.pdf`. Download generated files before ending the Colab session if you want to retain them.

The source folder did not include trained weights or a pinned dependency environment. The original notebook and saved outputs are preserved; training and reported results have not been rerun for this publication.

## Dataset attribution

The dataset files supplied with this project are labeled EmpatheticDialogues. The upstream dataset and benchmark are described in [Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset](https://arxiv.org/abs/1811.00207). See the [official EmpatheticDialogues repository](https://github.com/facebookresearch/EmpatheticDialogues) for the original dataset download, citation, and licensing information. The included CSV files are preserved as supplied with this project.

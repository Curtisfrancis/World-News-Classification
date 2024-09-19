# World-News-Classification

## Overview

This project focuses on classifying news articles into specific categories using machine learning. Efficient classification of data is essential for organizing and managing large datasets, helping improve data analysis and ensuring accuracy. The primary objective of this project is to build a model that accurately classifies news articles into one of five categories.

## Dataset

For this project, a public dataset from the BBC is used, containing 2,225 news articles. Each article is labeled under one of the following five categories:

- **Business**
- **Entertainment**
- **Politics**
- **Sport**
- **Tech**

### Dataset Split

- **Training Set**: 1,490 articles used to train the model.
- **Test Set**: 735 articles used to evaluate the model's performance.

## Objective

The primary objective is to develop a classification system that can categorize news articles based on their content. The system will be trained on the provided dataset and will aim to classify unseen articles into one of the five categories with high accuracy.

## Features

- **Text Preprocessing**: Clean and prepare the text data for model training.
- **Modeling**: Implement machine learning algorithms (using transformer-based models) to classify news articles.
- **Evaluation**: Assess the model's performance on the test dataset using various metrics like accuracy, precision, recall, and F1-score.

## Requirements

- **Python 3.8+**
- **Libraries**:
  - `pandas`
  - `scikit-learn`
  - `tensorflow`
  - `transformers`

You can install the necessary dependencies by running:

```bash
pip install pandas scikit-learn tensorflow transformers
```

## Model Architecture

This project utilizes the **DistilBERT** architecture, a smaller and faster variant of the BERT transformer model. DistilBERT is well-suited for tasks like text classification while maintaining a good balance between speed and accuracy.

### Steps:

1. **Data Splitting**: The dataset is split into training and test sets using stratified sampling.
   
2. **Tokenization**: The text data is tokenized using `DistilBertTokenizerFast` to convert the text into a format understandable by the model.

3. **Model Training**: Fine-tuning the pre-trained DistilBERT model for the task of sequence classification.

### Code Snippet for Model Training

```python
from transformers import TFTrainer, TFDistilBertForSequenceClassification, TFTrainingArguments

training_args = TFTrainingArguments(
    output_dir='./Outputs',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.1,
)

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
```

## Evaluation

The trained model is evaluated using the test set to assess its performance. Metrics such as accuracy, precision, recall, and F1-score are calculated to measure the model's ability to classify news articles correctly.

```python
from sklearn.metrics import classification_report

y_pred = trainer.predict(test_dataset).predictions.argmax(-1)
target_names = ['business', 'entertainment', 'politics', 'sport', 'tech']

print(classification_report(y_test, y_pred, target_names=target_names))
```

## Results

The classification report provides detailed metrics for each category, including precision, recall, and F1-score, which help evaluate the performance of the classification system.

Sample Output:
```
              precision    recall  f1-score   support

    business       0.85      0.88      0.86        67
entertainment       0.80      0.78      0.79        55
    politics       0.79      0.75      0.77        60
       sport       0.90      0.92      0.91        69
        tech       0.85      0.82      0.83        55

    accuracy                           0.84       306
   macro avg       0.84      0.83      0.83       306
weighted avg       0.84      0.84      0.84       306
```


## Conclusion

This project successfully demonstrates the use of a transformer-based architecture (DistilBERT) for classifying news articles. Further improvements can be made by:
- Increasing the number of training epochs.
- Experimenting with different learning rates and weight decay.
- Utilizing more advanced text preprocessing techniques.

Feel free to explore the model and adjust parameters to achieve higher accuracy or faster training times!

---

Let me know if you'd like to add or modify any sections!

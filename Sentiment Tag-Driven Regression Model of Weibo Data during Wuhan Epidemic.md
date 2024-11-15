# **Sentiment Tag-Driven Prediction Model of Weibo Data during Wuhan Epidemic**

## I. Introduction

This study constructs a Sentiment Analysis model with Weibo data during the Wuhan epidemic. The core techniques include TF-IDF vectorization, BernoulliNB, LinearSVC, and Logistic Regression. The goal is to automatically categorizing new Weibo post sentiment as Positive or Negative. 

Significantly, it provides access to public sentiment, benefiting policymakers and researchers in marketing, social studies, and journalism. Our group is specifically interested in using machine learning to explore the link between the epidemic and sentiment.

## II. Data Description

Our data was extracted from Weibo from 2019.12.30 to 2020.05.30 and contains 201,883 post records after cleaning. 

#### **It contains the following 6 fields:**

1. ids: the id of the post
2. text: the text of the post
3. sentiment_label: the polarity of the post (0 = Negative, 1 = positive)
4. sentiment_key: the polarity of the post
5. positive_probs: how much probability the post's polarity is likely to be positive
6. negative_probshow much probability the post's polarity is likely to be negative

We require only the **sentiment** and **text** fields, so we discard the rest.

#### Head

Note that we sort the data that negative records are positioned before positive records. 

|      |                                                         text | sentiment_label | sentiment_key |
| ---: | -----------------------------------------------------------: | --------------: | ------------- |
|    0 | 【感染诺如病毒该如何应对？】诺如病毒是一组形态相似、抗原性略有不同的病毒颗粒，主要表现为腹泻... |               0 | negative      |
|    1 | #坚决打赢疫情防控阻击战#【最新！蚌埠新增确诊病例1例情况公布】根据安徽省卫健委发布的最新疫... |               0 | negative      |
|    2 | #武汉新型冠状病毒肺炎疫情防控# 【#钻石公主号中国同胞将乘包机返港#】2月19日起，“钻石... |               0 | negative      |
|    3 | #武汉新型冠状病毒肺炎疫情防控# 【人社部引导农民工返工：#出家门上车门下车门进厂门#】人社... |               0 | negative      |

#### Distribution of labels

![image-20241109224938967](/Users/suihua/Library/CloudStorage/OneDrive-个人/ISTM/TERM1/DOTE5110 Statistical Analysis/Project/img/image-20241109224938967.png)

## III. Data Preprocessing

To render the text in a more digestible format for the machine learning models, the following preprocessing procedures are implemented:

1. **URL Replacement**: Hyperlinks commencing with **"http" or "https" or "www"** are supplanted with **"提到某链接"**.
2. **Username Replacement**: @Usernames are substituted with the term **"某用户"**. (For example: "@Kaggle" is converted to "USER")
3. **Stopword Removal**: Stopwords are those that contribute minimally to the semantic essence of a sentence and can be disregarded without impairing the overall meaning. We employ jieba to segment Chinese sentences and eliminate stopwords in accordance with a predefined list.

Now that we have attained the preprocessed dataset, we are afforded a more lucid perspective. We will proceed to generate a word cloud for both positive and negative Weibo posts. 

### Word-Cloud for Negative Posts

![image-20241109225335453](/Users/suihua/Library/CloudStorage/OneDrive-个人/ISTM/TERM1/DOTE5110 Statistical Analysis/Project/img/image-20241109225335453.png)

### Word-Cloud for Positive Posts

![image-20241109225341541](/Users/suihua/Library/CloudStorage/OneDrive-个人/ISTM/TERM1/DOTE5110 Statistical Analysis/Project/img/image-20241109225341541.png)

### Dealing with skewness

Note that our dataset is skewed, which brings potential risks. An unequal distribution of positive and negative samples may cause models to favor the majority class, leading to inaccurate evaluation and overlooking the minority sentiment, thus distorting the representation of public opinion.

To address this, we used an undersampling method. As the dataset was randomized and sorted by sentiment_label, we took the first 86,242 entries to balance the classes. 

## IV. Model Building and Evaluation

### A. Model Selection

Three different types of models are included in this study:

1. Bernoulli Naive Bayes (BernoulliNB)：Based on Bayes' theorem and the feature independence assumption, it is suitable for discrete data. The calculation is simple and efficient, which make it has an advantage for large, high-dimensional Weibo data.
2. Linear Support Vector Classification (LinearSVC)：It finds a hyperplane for classification and maps to a high-dimensional space to make data linearly separable. It performs excellently in handling linear or approximately linear data, is good at processing high-dimensional data, has strong robustness and excellent generalization.
3. Logistic Regression (LR)：It uses a logistic function to obtain the probability of a category. It is simple, easy to explain and can output probabilities, widely used in many classification tasks. It has stable and accurate performance in classification.

### B. Preparing for Training and Testing Data

The Preprocessed Data is divided into 2 sets of data:

- **Training Data:** The dataset upon which the model would be trained on. Contains 95% data.
- **Test Data:** The dataset upon which the model would be tested against. Contains 5% data.

#### TF-IDF Vectoriser

TF-IDF indicates what the importance of the word is in order to understand the document or dataset, it converts a collection of raw documents to a matrix of TF-IDF features. 

### C. Evaluation Metrics

Since our dataset is skewed, that is, the number of positive and negative samples is unbalanced, we will reduce the sample quantity of positive records with Undersampling, simply by cutting all records after index 86242. Accuracy, Precision, Recall, and F1 - score are taken as the main evaluation metrics.
$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Furthermore, we're plotting the **Confusion Matrix** to get an understanding of how our model is performing on both classification types.

### D. Model Training and Evaluation Results

#### Model 1: Bernoulli Naive Bayes

|      | precision | recall | f1-score | support |
| ---- | --------- | ------ | -------- | ------- |
| 0    | 0.92      | 0.80   | 0.85     | 2152    |
| 1    | 0.82      | 0.93   | 0.87     | 2161    |

| accuracy     | -    | -    | 0.86 | 4313 |
| ------------ | ---- | ---- | ---- | ---- |
| macro avg    | 0.87 | 0.86 | 0.86 | 4313 |
| weighted avg | 0.87 | 0.86 | 0.86 | 4313 |

![image-20241109231826503](/Users/suihua/Library/CloudStorage/OneDrive-个人/ISTM/TERM1/DOTE5110 Statistical Analysis/Project/img/image-20241109231826503.png)

#### **Model 2: Linear Support Vector Classification**

|      | precision | recall | f1-score | support |
| ---- | --------- | ------ | -------- | ------- |
| 0    | 0.93      | 0.93   | 0.93     | 2152    |
| 1    | 0.93      | 0.93   | 0.93     | 2161    |

| accuracy     | -    | -    | 0.93 | 4313 |
| ------------ | ---- | ---- | ---- | ---- |
| macro avg    | 0.93 | 0.93 | 0.93 | 4313 |
| weighted avg | 0.93 | 0.93 | 0.93 | 4313 |

![image-20241109231832801](/Users/suihua/Library/CloudStorage/OneDrive-个人/ISTM/TERM1/DOTE5110 Statistical Analysis/Project/img/image-20241109231832801.png)

#### **Model 3: Logistic Regression**

|      | precision | recall | f1-score | support |
| ---- | --------- | ------ | -------- | ------- |
| 0    | 0.92      | 0.92   | 0.92     | 2152    |
| 1    | 0.92      | 0.92   | 0.92     | 2161    |

| accuracy     | -    | -    | 0.92 | 4313 |
| ------------ | ---- | ---- | ---- | ---- |
| macro avg    | 0.92 | 0.92 | 0.92 | 4313 |
| weighted avg | 0.92 | 0.92 | 0.92 | 4313 |

![image-20241109231839840](/Users/suihua/Library/CloudStorage/OneDrive-个人/ISTM/TERM1/DOTE5110 Statistical Analysis/Project/img/image-20241109231839840.png)

We can clearly see that the Linear Support Vector Classification performs the best out of all the different models that we tried. It achieves nearly **93% accuracy** while classifying the sentiment of a tweet.

Although it should also be noted that the **BernoulliNB Model** is the fastest to train and predict on. It also achieves 86% accuracy while calssifying.

## V. Model Application

Using AI to generate three news, respectively negative, positive and negative, and apply the pre-trained model to classify them. All three test cases passed successfully. 

## VII. Appendix

数据来源:公众号“月小水长”

https://pan.baidu.com/s/10eM4wf5Wqo8jHwANEzIP9g

停用词表

https://github.com/goto456/stopwords

模型文件

测试用例
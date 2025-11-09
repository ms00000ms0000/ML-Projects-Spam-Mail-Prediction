# 📬 ML Project - Spam Mail Prediction  

## 🚀 Overview  
This **Machine Learning project** is a **Spam Mail Prediction System** developed using the **Logistic Regression** algorithm.  
The dataset consists of two columns — **Category (Spam/Ham)** and **Message**.  

Text data was transformed into numerical features using **TF-IDF Vectorizer** for effective feature extraction.  
The model was trained to classify emails as spam or legitimate based on message content and achieved an impressive **accuracy of 96%**.  

This project showcases the use of **Natural Language Processing (NLP)** techniques for **text classification** and demonstrates how ML can enhance **email filtering systems**, improving security and reducing unwanted messages in digital communication.  

---

## 🔍 About the Project  
The **Spam Mail Prediction System** applies **supervised machine learning** to detect and classify emails as **spam** or **ham (not spam)**.  
It uses **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization to convert textual data into numerical form suitable for classification.  

This project highlights practical implementation of **NLP and text mining**, essential for modern **email filtering**, **phishing prevention**, and **content moderation systems**.  

---

## 🧠 Model Architecture  
The project uses the **Logistic Regression** algorithm, a simple yet powerful linear model for binary classification problems.  

* **Algorithm:** Logistic Regression  
* **Problem Type:** Binary Classification  
* **Feature Extraction:** TF-IDF Vectorization  
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

---

## 🧾 Dataset Description  
The dataset contains email messages labeled as spam or ham for supervised learning.  

| Feature | Description |
|----------|-------------|
| **Category** | Target label — "spam" or "ham" |
| **Message** | Text content of the email/message |

---

## ⚙️ Tech Stack & Libraries  

**Language:**  
* Python 🐍  

**Libraries:**  
* **NumPy** – Numerical computations  
* **Pandas** – Data handling and preprocessing  
* **Scikit-learn** – Logistic Regression, TF-IDF Vectorizer, and evaluation metrics  
* **Matplotlib / Seaborn** – Visualization of confusion matrix and results  

---

## 🚀 Features  
* Detects **spam** and **non-spam (ham)** messages with high accuracy  
* Uses **TF-IDF Vectorization** for effective text feature extraction  
* Implements **Logistic Regression** for binary classification  
* Achieves **96% model accuracy** on test data  
* Demonstrates real-world **NLP and spam filtering** application  

---

## 📊 Results  
The **Logistic Regression** model achieved an impressive **accuracy of 96%**, showing excellent performance in differentiating spam from legitimate messages.  
It provides a **lightweight, fast, and scalable** solution suitable for integration with real-world email systems.  

---

## 📁 Repository Structure  

```
📦 ML-Projects-Spam-Mail-Prediction
│
├── Spam_Mail_Prediction.ipynb                                            # Complete model implementation
├── spammail.csv                                                          # Dataset used for training and testing
└── README.md                                                             # Project documentation
```

---

## 🧪 How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/ms00000ms0000/ML-Projects-Spam-Mail-Prediction.git
   cd ML-Projects-Spam-Mail-Prediction


2. **Install dependencies:**
    ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
    ```bash
   jupyter notebook Spam_Mail_Prediction.ipynb
   ```

4. **Execute all cells to train, test, and evaluate the model.**

---

## 📈 Future Improvements

* Add Naive Bayes and SVM models for performance comparison

* Implement a Streamlit web interface for real-time spam detection

* Include advanced NLP preprocessing (stopword removal, stemming, lemmatization)

* Deploy model as a REST API for integration with email clients

---

## 👨‍💻 Developer

Developed by: Mayank Srivastava

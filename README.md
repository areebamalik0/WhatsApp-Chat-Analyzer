# WhatsApp Chat Analyzer

## **Overview**

WhatsApp Chat Analyzer is an AI-powered application developed in Python that analyzes exported WhatsApp chats and provides meaningful insights about conversations. The application supports direct messages (DMs), group chats, and communities exported as `.txt` files.

Using Machine Learning and Natural Language Processing (NLP), the application performs toxicity detection, sentiment analysis, and chat activity analysis through an interactive Streamlit interface.

---

## **Features**

* Most active user detection
* Most frequently used word analysis
* Most frequently used emoji analysis
* Toxicity percentage calculation
* Non-toxicity percentage calculation
* Sentiment analysis

  * Positive messages percentage
  * Negative messages percentage
  * Neutral messages percentage
* Graphical visualization of chat statistics
* Interactive web interface built with Streamlit

---

## **Technologies Used**

* Python
* Streamlit
* Scikit-learn
* NLTK
* Pandas
* Matplotlib
* Joblib
* Emoji

---

## **Project Structure**

```text
WhatsApp Chat Analyzer/
│
├── app.py
├── toxicity_model.py
├── whatsapp.png
│
├── csv files/
│   └── train.csv
│
├── toxicity_model.pkl        (Generated after training)
├── toxicity_vectorizer.pkl   (Generated after training)
│
└── README.md
```

---

## **Setup Instructions**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
```

### 2. Install Required Libraries

```bash
pip install pandas nltk joblib scikit-learn matplotlib emoji streamlit
```

### 3. Update File Paths

Before running the project, update the file paths in `toxicity_model.py`.

Replace:

```python
C:/Users/SystemUsername/Desktop/WhatsApp Chat Analyzer/
```

with the location where you have stored the project on your system.

Also ensure that the path to:

```text
csv files/train.csv
```

matches its location on your computer.

---

## **Training the Toxicity Model**

Run the following command:

```bash
python toxicity_model.py
```

This will generate:

* `toxicity_model.pkl`
* `toxicity_vectorizer.pkl`

These files are required for the application to function correctly.

---

## **Running the Application**

Start the Streamlit application using:

```bash
streamlit run app.py
```

After the application launches in your browser:

1. Export a WhatsApp chat as a `.txt` file.
2. Upload the exported chat.
3. View the generated analysis and visualizations.

---

## **Supported Language**

Currently, the application supports **English-language chats only**.

Chats containing mostly non-English text may not produce accurate results because the toxicity and sentiment analysis models are designed for English text.

---

## **Future Enhancements**

* Multi-language support
* Word cloud generation
* Additional chat statistics
* Improved toxicity classification
* Advanced visual analytics

---

## **License**

This project is licensed under the MIT License.

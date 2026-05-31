# WhatsApp Chat Analyzer

## **Overview**

WhatsApp Chat Analyzer is an AI-based application written in Python that analyzes exported WhatsApp chats and provides useful insights about conversations. The application supports direct messages (DMs), group chats, and communities exported as `.txt` files.

The project uses a Machine Learning model for toxicity detection and Natural Language Processing (NLP) techniques for sentiment analysis.

---

## **Features**

* Identify the most active user in the chat
* Find the most frequently used word
* Find the most frequently used emoji
* Calculate toxicity percentage
* Calculate non-toxicity percentage
* Perform sentiment analysis

  * Positive percentage
  * Negative percentage
  * Neutral percentage
* Display graphical representations of the analysis
* User-friendly interface built with Streamlit

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
├── toxicity_model.pkl        (Generated after training)
├── toxicity_vectorizer.pkl   (Generated after training)
└── README.md
```

---

## **Important Note**

The repository does **not** include:

* `toxicity_model.pkl`
* `toxicity_vectorizer.pkl`
* Training dataset (`train.csv`)

Users must train the model themselves by running `toxicity_model.py`.

Before running the project, replace all file paths in `toxicity_model.py` with the correct paths on your system.

Example:

```python
C:/Users/SystemUsername/Desktop/WhatsApp Chat Analyzer/
```

Replace `SystemUsername` with your own Windows username and adjust the remaining path according to where your project files are stored.

You must also update the path of `train.csv` to match its location on your system.

---

## **Installation**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
```

### 2. Install Required Libraries

```bash
pip install pandas nltk joblib scikit-learn matplotlib emoji streamlit
```

### 3. Download the Dataset

Download the toxicity dataset and update its path inside `toxicity_model.py`.

---

## **Training the Model**

Run:

```bash
python toxicity_model.py
```

This will generate:

* `toxicity_model.pkl`
* `toxicity_vectorizer.pkl`

These files must be present in the project folder before running the application.

---

## **Running the Application**

Run:

```bash
streamlit run app.py
```

Then upload an exported WhatsApp `.txt` chat file to begin analysis.

---

## **Supported Language**

Currently, the application supports **English-language chats only**.

Chats containing mostly non-English text may not produce accurate results.

---

## **Future Improvements**

* Multi-language support
* Word cloud generation
* Additional chat statistics
* Improved toxicity classification
* Enhanced visualizations

---

## **License**

This project is released under the MIT License.

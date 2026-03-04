# 🏢 Nikhil's Workforce Intelligence Suite

A Streamlit dashboard demonstrating **Classification, Clustering, and Association Rule Mining** working together to solve a real-world COVID-era workforce planning crisis — from **Descriptive to Prescriptive analytics**.

---

## 📖 The Business Scenario

**2019:** Nikhil founds a software development company with 100 employees (developer-heavy mix).

**2020 — COVID hits:** A client, Ritika, offers a survival lifeline — develop 2 software products in 3 months. **Condition:** Every assigned employee must know **Python + English + French**.

**The Problem:** Nikhil needs 30 employees (+ 15-day buffer). HR discovers **only 3 of 100 employees** have all 3 skills. He must find and train **37 more** (27 gap + 10 buffer) from the remaining 97.

**The Analytics Challenge:** Use data analytics to (1) find the best 37 trainable candidates, (2) predict if they'll leave during the project, and (3) discover hidden patterns in workforce data.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Dashboard Structure (7 Tabs)

| Tab | Analysis Type | ML Technique |
|---|---|---|
| 📖 The Story | Business Context | — |
| 📊 Descriptive | What happened? | — |
| 🔍 Diagnostic | Why did it happen? | Statistical Tests |
| 🎯 Clustering | Segment employees | K-Means |
| 🤖 Classification | Predict attrition | LR, RF, GBM |
| 🔗 Association Rules | Discover patterns | Apriori |
| 💊 Prescriptive | What should we do? | All combined |

---

## 🛠️ Tech Stack

Streamlit · Plotly · scikit-learn · SciPy · mlxtend (Apriori) · Pandas · NumPy

---

## 📦 Dataset

100 employees × 31 columns: demographics, 5 technical skills, 3 languages, performance metrics, satisfaction scores, and attrition risk.

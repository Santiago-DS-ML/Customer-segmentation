## 📌 Project Overview

In retail and customer relationship management (CRM), treating all consumers uniformly leads to inefficient marketing spend and sub-optimal engagement. This project addresses this problem by architecting a full data science pipeline that:
1. Engages in rigorous Exploratory Data Analysis (EDA) on raw consumer demographics.
2. Formulates an optimal feature engineering strategy, including dimensionality reduction via **Principal Component Analysis (PCA)**.
3. Benchmarks four distinct clustering paradigms (**K-Means**, **HAC**, **DBSCAN**, **HDBSCAN**).
4. Deploys a champion model to isolate cohesive consumer archetypes.
5. Dynamically interprets mathematical clusters into production-ready business strategies via **Google Gemini**.

## 📁 Repository Structure


```

```text
README.md created successfully.

```bash
├── Customer_segmentation.ipynb   # Complete development & benchmarking notebook
├── Customer_Segmentation_Report.pdf # Comprehensive technical and executive report
├── README.md                      # Production-ready documentation
└── requirements.txt               # Project dependency matrix

```

---

## 🛠️ Data Architecture & Engineering

### 1. Dataset Profile

The analysis uses the benchmark **Mall Customers Dataset** (Kaggle: `vjchoudhary7/customer-segmentation-tutorial-in-python`).

* **Sample Size:** $N = 200$ rows, completely clean ($0$ missing values, $0$ duplicates).
* **Core Dimensions:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`, and `Gender`.

### 2. Feature Pipeline

* **Categorical Encoding:** `Gender` converted into an exact binary indicator ($0$ for Female, $1$ for Male).
* **Index Stripping:** Structural identifiers (`CustomerID`) were dropped to prevent distance matrix skew.
* **Dimensionality Reduction:** Implemented **PCA (Principal Component Analysis)** with whitening. The first two principal components successfully capture **~90% of the global variance**, allowing for clean 2D spatial visualization.

---

## 🔬 Modeling Framework & Benchmarking

To ensure the highest cluster cohesion and separation, four algorithmic variations were built and evaluated:

| Model Architecture | Key Hyperparameters | Structural Behavior / Results |
| --- | --- | --- |
| **Variation 1: DBSCAN** | `eps=20`, `min_samples=20` | Highly volatile. Classified an excessive volume of standard records as noise. |
| **Variation 2: HDBSCAN** | `min_cluster_size=3` | Solid hierarchical density mapping, but boundaries remained too fluid for CRM assignment. |
| **Variation 3: HAC (Hierarchical)** | `linkage='ward'`, `n_clusters=3` | Clean macro-level grouping, but lacked operational granularity. |
| **Variation 4: K-Means (Champion)** | `n_clusters=3`, `init='k-means++'` | **Optimal**. Identified clear, distinct boundaries with minimal internal variance. |

### Mathematical Selection

The K-Means cluster count was optimized using the **Elbow Method** by tracking the Within-Cluster Sum of Squares (WCSS):

$$WCSS = \sum_{j=1}^{K} \sum_{i \in C_j} ||x_i - \mu_j||^2$$

---

## 💡 Key Findings & Strategic Personas

By passing the mathematical centroid profiles directly to **Google Gemini**, the pipeline automatically translates data clusters into operational business personas:

### 🟢 Cluster 1: The "Thrifty & Modest Earners"

* **Metrics:** Low-to-moderate income ($15k–$67k), low-to-moderate spending score ($3–$57). Broad age distribution.
* **Persona:** Value-conscious consumers who optimize price-to-utility ratio.
* **Action Plan:** Target with aggressive flash sales, bundle packaging (BOGO), and transaction-driven reward systems.

### 🔵 Cluster 2: The "Impulsive & Trend-Driven Consumers"

* **Metrics:** High spending score ($69–$99) across completely volatile income brackets ($15k–$137k). Youth-concentrated (ages 21–38).
* **Persona:** Hyper-engaged consumers highly responsive to trends, social content, and prestige mechanics.
* **Action Plan:** Implement premium tiered VIP programs, real-time personalized recommendations, and exclusive early-access product drops.

### 🟠 Cluster 3: The "High-Income, Low-Spend Affluents"

* **Metrics:** Massive financial power ($70k–$137k) paired with severely suppressed spending scores ($1–$40).
* **Persona:** A critical unmapped market opportunity. These buyers face friction or suffer from extreme brand detachment.
* **Action Plan:** Deploy qualitative feedback loops to isolate pain points, offer bespoke white-glove personalization, and extend private VIP event invitations.

---

## 🚀 Execution & Deployment

### 1. Installation & Environment Set Up

Clone the repository and install the production requirements matrix:

```bash
git clone [https://github.com/Santiago-DS-ML/customer-segmentation.git](https://github.com/Santiago-DS-ML/customer-segmentation.git)
cd customer-segmentation
pip install -r requirements.txt

```

### 2. Live LLM Integration

To dynamically reproduce the cluster synthesis and marketing blueprints, expose your Gemini API key inside your environment context:

```python
import google.generativeai as genai
from google.colab import userdata

# Configure API Key securely
genai.configure(api_key=userdata.get('GEMINI_API_KEY'))
model = genai.DiscriminativeModel('gemini-1.5-pro')

```

### 3. Running the Pipeline

Open the notebook file inside your Jupyter or Google Colab environment:

```bash
jupyter notebook Customer_segmentation.ipynb

```

---

## 📈 Limitations & Engineering Roadmap

* **Sample Expansion:** Scale the pipeline to ingest large-scale production enterprise records ($N > 100,000$).
* **RFM Transition:** Shift away from synthetic scores and ingest raw point-of-sale logs to build an organic **Recency, Frequency, Monetary (RFM)** engine.
* **Localized Supervised Extension:** Segment the production database using the K-Means champion model, then train independent local supervised regressors (e.g., Random Forest) on each cluster to predict upcoming purchase volumes, benchmarking performance against a single global estimator.

---

*Developed as part of an Advanced Data Science Capability Matrix.*
"""


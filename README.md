# Public Funding Amount Prediction
*This Python workflow predicts public funding amounts for renovation projects using semantic NLP embeddings and gradient boosting.*

---

## 🎯 Overview
This project develops a robust predictive model to estimate public funding amounts (`montant_engage`) allocated to energy renovation and territorial development projects.

**Objectives**
- Maximize the coefficient of determination ($R^2$) on unseen test data
- Implement a reproducible Data-Centric functional pipeline
- Ensure interpretability using SHAP values and feature importance

---

## 🗄️ Data
- **Source:** Kaggle Competition (M2 ECAP Course)
- **Time Period / Size:** 7,094 training rows, 1,774 test rows
- **Target Variable:** Committed Amount (`montant_engage`)
- **Key Predictors / Features:** Project description (`resume_projet`), Location (`commune_code`), Legal Category
- **Preprocessing:** Handled missing values via informative imputation, target encoding for high-cardinality features, FastText embeddings
- **Data Availability:** Provided in `data/`

---

## 🧠 Methodology
- **Theoretical Approach:** Functional modular architecture with strict train/test separation
- **Mathematical Framework:** Transformed Target Regression (Box-Cox/Yeo-Johnson) for skewed distributions
- **Evaluation Strategy:** Stratified K-Fold Cross-Validation (k=5/10) on quantile bins, RandomizedSearchCV

---

## ⚙️ Features
- **Build Data Pipeline**: Process end-to-end data securely via custom Orchestrator
- **Engineer Features**: Transform high-cardinality categorical data via Target Encoding
- **Embed Semantics**: Vectorize project summaries using pre-trained FastText French models
- **Optimize Models**: Tune hyperparameters via RandomizedSearchCV for Gradient Boosting (CatBoost, XGBoost)
- **Interpret Predictions**: Analyze marginal effects with SHAP and Partial Dependence Plots

---

## 🧰 Tech Stack
- **Language**: Python 3.13+
- **Data Engineering & Acquisition**: requests
- **Numerical Computing & Data Manipulation**: pandas, NumPy, SciPy
- **Machine Learning & Deep Learning**: scikit-learn, CatBoost, XGBoost, LightGBM, gensim
- **Data Visualization**: matplotlib, seaborn, shap
- **Reporting & Documentation**: JupyterLab, Notebook, IPython

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/master-year2-decision-trees.git
cd master-year2-decision-trees
uv sync
```

---

## 💻 Usage Example

### Reproducing the Analysis / Execution Pipeline
*(Expected runtime: ~3 hours on standard CPU)*

```bash
uv run jupyter notebook project.ipynb
```

---

## 📂 Project Structure

```text
master-year2-decision-trees/
│
├── data/
│   ├── test.csv                     # Evaluation dataset (Target hidden)
│   └── train.csv                    # Labeled training dataset
├── .gitignore
├── LICENSE
├── README.md
├── project.ipynb                    # End-to-end analysis and modeling pipeline
├── pyproject.toml
└── uv.lock
```

---

## 📈 Results

### Performance Metrics
| Model / Strategy | Complexity / Size (e.g. Assets/Variables) | R² (CV Mean) |
|------------------|-------------------------------------------|--------------|
| Baseline         | N/A                                       | 0.0000       |
| **CatBoost**     | **Depth=4, Iter=500**                     | **0.5319**   |

### Key Findings
- **Semantic Impact:** The FastText embedding dimension F221 proved to be the strongest predictor, capturing key renovation terms in project summaries.
- **Geographic Influence:** Location (`code_commune`) significantly determines funding levels due to local policy variations.

---

## 🚧 Limitations & Future Work
- **Data Enrichment:** Integrate external INSEE data (population, fiscal income) to capture territorial disparities
- **Meta-Modeling:** Combine uncorrelated predictions (Random Forest + CatBoost) via VotingRegressor
- **Advanced NLP:** Deploy CamemBERT transformers to capture finer semantic contexts

---

## 📜 License
This project is released under the MIT License.
© 2026 Florian Crochet

---

## 👤 Author
**Florian Crochet**  
[GitHub Profile](https://github.com/floriancrochet)

*Master 2 – Econometrics & Statistics, Applied Econometrics Track*

---

## 🤝 Acknowledgments
This work was conducted as part of the Machine Learning course, supervised by Cédric Dangeard.

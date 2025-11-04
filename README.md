# ml_25-26_audience_decode

**Audience Decode Project Breakdown**

---

## **1️⃣ OBTAIN – Get and understand the data**

**Goal:** connect, inspect and understand the structure of the dataset (`viewer_interactions.db`).

### Tasks

* Connect to the `.db` file using `sqlite3` or `duckdb`.
* List all available tables.
* For each table:

  * Inspect columns (name, type, PK, FK, nullable).
  * Count number of rows.
  * Check for relationships between tables (foreign keys).
* Build a **data dictionary** summarizing all tables and columns.
* Merge or identify the **main table** (`interactions`, `ratings`, etc.) for analysis.
* Save a summary table and small CSV sample for quick previews.

**Output:**
`data_dictionary.md`, `table_summary.csv`, small data sample for EDA.

---

## **2️⃣ SCRUB – Clean and preprocess**

**Goal:** prepare a clean, usable dataset for modeling.

### Tasks

* Handle **missing values** (drop, impute with mean/median/mode as appropriate).
* Handle **outliers** (remove or winsorize).
* Convert **timestamps** to datetime and extract useful features (month, year, weekday, hour).
* Encode **categorical variables** (label, one-hot, or target encoding).
* Normalize / scale numerical variables.
* Create **train / validation / test splits**:

  * Train / validation used for model tuning.
  * Test only used at the end (final evaluation).

**Output:**
`X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` (if supervised)
or clean feature matrix (if clustering).
`preprocess_pipeline.py` or equivalent.

---

## **3️⃣ EXPLORE – EDA and pattern discovery**

**Goal:** understand audience behavior and patterns.

### Tasks

* Perform **Exploratory Data Analysis (EDA)**:

  * View distributions (ratings, genres, time spent, etc.).
  * Identify temporal trends (view counts per month).
  * Find top users, top movies, most common genres.
  * Check correlations between numerical variables.
  * Visualize missing data and outliers.
* Build **basic visualizations**:

  * Histograms, boxplots, bar charts.
  * Time-series trends.
  * Heatmaps (correlation matrix).
* Formulate hypotheses:

  * Are there clusters of similar users?
  * How does engagement change over time?
  * Are certain genres more popular for specific user types?

**Output:**
EDA notebook (`01_EDA.ipynb`)
EDA summary report (`eda_summary.md`) with 4–5 clear insights and visualizations.

---

## **4️⃣ MODEL – Build, test and compare models**

**Goal:** model audience behavior (not only predict ratings) through segmentation and pattern discovery.

### Tasks

* Define **the main goal**:

  * This is a **clustering problem** (group users by viewing patterns, preferences, and engagement).
* Select and test **at least 3 models**:

  * For clustering:

    * `KMeans`
    * `Gaussian Mixture Model (GMM)`
    * `DBSCAN` or `Agglomerative Clustering`
  * Optionally add a **supervised model** to predict rating or engagement (e.g. `RandomForest`, `XGBoost`).
* Perform **cross-validation or hyperparameter tuning**:

  * Tune number of clusters (k).
  * Tune model-specific hyperparameters (distance metric, epsilon, components, etc.).
* Evaluate models using correct metrics:

  * For clustering: **Silhouette score**, **Davies–Bouldin**, **Calinski–Harabasz**.
  * For regression/classification (optional): **RMSE**, **MAE**, **AUC**, **Accuracy**.
* Select the **best model** and justify your choice.

**Output:**
`model_comparison.csv` with metrics and parameters
`03_Modeling.ipynb` notebook with plots (elbow, silhouette, etc.)

---

## **5️⃣ iNTERPRET – Explain, visualize and report**

**Goal:** turn models into insights that can inform business strategy.

### Tasks

* **Describe the clusters** (profiling):

  * Mean rating, activity level, favorite genres, engagement frequency.
  * Label clusters with descriptive names (e.g. “Casual Watchers”, “Genre Enthusiasts”, “Binge Viewers”).
* **Interpret model outputs**:

  * Visualize clusters (e.g. PCA 2D plot or t-SNE).
  * Show feature importance or centroid values.
* **Fairness and interpretability check**:

  * Are some user groups underrepresented?
  * Is the segmentation stable across time or demographics?
* Summarize findings:

  * What behaviors define each cluster?
  * How could this help **content curation** or **recommendation**?

**Output:**
`reports/cluster_profiles.md`, `reports/executive_summary.md`, and all visuals in final notebook.

---

## **6️⃣ COMMUNICATE – Deliverables & Documentation**

**Goal:** document everything clearly for reproducibility.

### Tasks

* Write a **README.md** (project overview + how to run + results summary).
* Include:

  * Environment and dependencies.
  * Folder structure.
  * Description of chosen models and why.
  * Key visualizations or metrics.
* Export best model(s) and pipeline(s) for reuse.

**Output:**
✅ `README.md`
✅ Final notebooks (EDA, Features, Modeling)
✅ Reports folder with visuals and insights
✅ Optional `environment.yml` or `requirements.txt`

---

**Final Deliverables Overview**

| Category       | Deliverable                | Format           |
| -------------- | -------------------------- | ---------------- |
| Data audit     | Table & schema summary     | `.csv`, `.md`    |
| Preprocessing  | Clean dataset & pipeline   | `.py`, `.pkl`    |
| EDA            | Plots & insights           | `.ipynb`, `.md`  |
| Modeling       | Comparison & best model    | `.ipynb`, `.csv` |
| Interpretation | Cluster profiles & visuals | `.md`, `.png`    |
| Documentation  | Final README & setup       | `.md`, `.yml`    |


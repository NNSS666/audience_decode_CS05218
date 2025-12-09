
# 4. Clustering Analysis — Summary of All Steps Performed

This section summarizes the entire clustering workflow, from feature engineering to model selection, validation, and behavioral interpretation.  
The objective was to identify meaningful user segments based on engagement patterns, rating behavior, recency, and movie-preference signals.

---

## 4.1 Feature Engineering for User-Level Clustering

User-level aggregated features were generated to capture essential behavioral dimensions:

### **Engineered Features**
- **scaled_total_ratings** — Standardized version of total_ratings.
- **activity_days_new** — Duration (in days) between first and last recorded rating.
- **days_since_last_interaction** — Recency relative to the most recent date in the dataset.
- **ratings_per_active_day** — Rating intensity (total_ratings / activity_days_new).
- **avg_rating_from_customer** — Mean rating value issued by the user.
- **std_rating_from_customer** — Rating variance, measuring consistency.
- **avg_movie_popularity** — Average movie popularity (after standardizing total_ratings_movie at the interaction level).

### **Missing-Value Handling**
Mean imputation was applied to:
- avg_rating_from_customer  
- std_rating_from_customer  
- avg_movie_popularity  

These seven features form the final clustering feature set:

```

scaled_total_ratings
activity_days_new
days_since_last_interaction
ratings_per_active_day
avg_rating_from_customer
std_rating_from_customer
avg_movie_popularity

```

---

## 4.2 Data Standardization

Since the feature scales differ substantially, the entire feature matrix was standardized using:

**StandardScaler → X_scaled**

This ensures that distance-based methods (KMeans, GMM) and density-based methods (HDBSCAN) properly capture meaningful behavioral differences rather than raw magnitude differences.

---

## 4.3 Dimensionality Reduction with PCA

PCA was used for visualization and to support density-based clustering:

- **PCA-2** explained ~47% of the variance → useful for visual inspection.
- **PCA-5** explained ~69% of the variance → used as input for HDBSCAN for improved performance and stability.

---

## 4.4 KMeans Clustering (k = 2–10)

### **Metrics Evaluated**
- **Inertia (SSE)** — Elbow curve (not very clear in this dataset).
- **Silhouette Score** — Primary metric for model selection.

### **Key Results**
- Best silhouette values:
  - **k = 7 → ~0.222**
  - **k = 4 → ~0.202**
  - **k = 10 → ~0.199**

**k = 7** was chosen as the best KMeans solution because:
- It provides a good balance between granularity and separation.
- Cluster profiles are interpretable.
- No cluster is too small or dominated by noise.

---

## 4.5 Gaussian Mixture Models (GMM)

GMM was evaluated for 2–10 components using silhouette.

### **Results**
- Highest silhouette at **k = 2 (≈ 0.245)**.
- However, Gaussian assumptions fail due to:
  - heavy-tailed distributions,  
  - extreme rating consistency (std ≈ 0),  
  - irregular density structure,  
  - large variance differences across features.

### **Conclusion**
Although GMM achieves a high silhouette at k=2, it produces broad, non-interpretable clusters and does not capture the true behavioral structure.  
**GMM was discarded.**

---

## 4.6 HDBSCAN Clustering

HDBSCAN is well suited for:
- non-spherical clusters  
- highly imbalanced densities  
- natural behavioral patterns  
- presence of noise/outliers  

### **Grid Search**
Parameters tested:
- **min_cluster_size**: 2000, 4000, 8000  
- **min_samples**: 10, 30, 50

### **Evaluation Metrics**
- number of clusters  
- percentage of noise  
- silhouette score (on non-noise points)  
- largest/smallest cluster size  

### **Best Configuration**
- **min_cluster_size = 8000**  
- **min_samples = 30**  
- silhouette ≈ **0.275**  
- noise ≈ **20.8%**  
- clusters identified: **6**

### **Interpretation**
HDBSCAN produced:
- highly compact and well-separated clusters  
- meaningful behavioral profiles  
- correct identification of ~20% noise (unclusterable users)

HDBSCAN clearly outperforms KMeans and GMM in structural clarity, interpretability, and density alignment.

---

## 4.7 Cluster Profiling

Each cluster was analyzed using:
- feature means  
- feature standard deviations  
- size of each cluster  
- rating behavior  
- engagement levels  
- recency  
- movie popularity preference  

### **Main Findings**
HDBSCAN revealed natural, interpretable user types, including:

- ultra-short-lived harsh reviewers  
- niche-oriented dormant users  
- neutral short-term users  
- moderately active long-term positive users  
- highly engaged, diverse long-term users  

KMeans produced similar groups but with lower consistency and more mixing of behaviors.

---

## 4.8 Model Comparison Summary

| Method | Advantages | Limitations | Outcome |
|--------|------------|-------------|---------|
| **KMeans** | Simple, stable, interpretable; decent silhouette | Assumes spherical clusters; forces all users into a segment | Good (k=7) |
| **GMM** | Soft clustering, probabilistic | Assumes Gaussian shapes; fails on heavy-tailed behavioral data | Discarded |
| **HDBSCAN** | Captures natural density structure; identifies noise; best silhouette; highly interpretable | Does not assign all users; parameter sensitivity | **Best performing model** |

### **Final Recommendation**
**HDBSCAN provides the most meaningful, stable, and behaviorally interpretable clustering solution.**  
It identifies clear behavioral groups while correctly excluding irregular or low-information users as noise.

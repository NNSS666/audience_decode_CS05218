# **4. Clustering Analysis**

## **4.1 Feature Engineering and Data Preparation**

The first step was to transform raw interaction logs into a meaningful set of user-level behavioral indicators. To understand differences in user engagement and rating style, we aggregated interaction data per customer and constructed seven key features.

These features were designed to capture **how much a user interacts**, **how long they stay active**, **how recently they were engaged**, **how they rate items**, and **the popularity of the content they interact with**. After converting these raw measures into standardized versions where necessary, the final feature set included:

* **scaled_total_ratings** → a standardized measure of overall rating volume
* **activity_days_new** → duration between a user's first and last rating
* **days_since_last_interaction** → how long it has been since a user last engaged
* **ratings_per_active_day** → frequency and intensity of rating behavior
* **avg_rating_from_customer** → general positivity or negativity in rating style
* **std_rating_from_customer** → rating consistency or variability
* **avg_movie_popularity** → tendency toward mainstream vs. niche content

Missing values were handled using simple mean imputation for rating-based statistics and movie-popularity features, ensuring a clean, complete dataset for clustering.

This set of seven features served as the foundation for all subsequent modeling.

## **4.2 Standardization and Dimensionality Reduction**

Because the engineered features operated on very different scales, the dataset was standardized using `StandardScaler`. This step ensured that models relying on distances or density estimation would not be biased by raw magnitudes.

To explore the structure of the data and assist visualization, PCA was applied.

* A **2-component PCA** captured ~47% of the total variance, enough to visualize broad patterns and potential cluster separations.
* A **5-component PCA** captured ~84% of the variance, providing a more stable low-dimensional space that density-based methods like HDBSCAN could use effectively.

PCA therefore played both an exploratory and a practical role in improving clustering performance.

## **4.3 KMeans Clustering (k = 2–10)**

KMeans was used as a baseline model due to its simplicity and interpretability. We ran the algorithm for values of k ranging from 2 to 10, evaluating each configuration through inertia curves and silhouette scores.

While the inertia curve did not reveal a clear elbow point, the silhouette analysis provided more actionable insights. The strongest performers were:

* **k = 7** (silhouette ≈ 0.222)
* **k = 4** (silhouette ≈ 0.221)
* **k = 10** (silhouette ≈ 0.218)

Among these, **k = 7** offered the best compromise between behavioral detail and cluster separation. The resulting clusters were reasonably interpretable and relatively balanced in size, although some mixing of behaviors was still present due to the algorithm’s assumption of spherical and equally dense clusters.

## **4.4 Gaussian Mixture Models (GMM)**

We then tested Gaussian Mixture Models, which provide soft assignments and can theoretically adapt to more flexible cluster shapes. Silhouette scores suggested that **k = 2** delivered the best result (≈ 0.245). However, this was misleading.

The underlying behavioral distributions are highly skewed, heavy-tailed, and vary greatly in density. These characteristics violate GMM’s Gaussian assumptions. As a result, although the model numerically achieved a good silhouette score, the resulting clusters were too broad and lacked behavioral meaning. They failed to differentiate between distinct patterns of engagement and rating behavior.

For this reason, despite promising initial metrics, **GMM was rejected**.

## **4.5 HDBSCAN: Density-Based Clustering**

Given the limitations of KMeans and GMM, we turned to HDBSCAN, a density-based algorithm capable of identifying clusters of arbitrary shape and distinguishing dense groups from sparse noise.

A grid search was conducted using various combinations of `min_cluster_size` (2000, 4000, 8000) and `min_samples` (10, 30, 50). Each configuration was evaluated based on:

* number of clusters
* silhouette score (computed only on non-noise points)
* proportion of noise
* size of largest and smallest clusters

The most effective setting was:

* **min_cluster_size = 8000**
* **min_samples = 30**

This configuration produced **6 well-defined clusters**, with a silhouette score of ~0.275 and a noise level of ~20%. The noise points represented users whose behavior does not fit any consistent pattern—an important advantage of HDBSCAN over KMeans, which forces every user into a cluster even when their behavior is ambiguous.

The behavioral structure uncovered by HDBSCAN was significantly clearer and more coherent than what was achievable with KMeans or GMM.

## **4.6 Cluster Profiling and Behavioral Interpretation**

For all models, cluster profiles were analyzed through feature means and variances, enabling us to characterize each user segment.

HDBSCAN in particular produced clusters that aligned strongly with real behavioral patterns. Examples include:

* short-lived, harsh reviewers
* long-term users with consistent positivity
* moderately active users with stable habits
* dormant users resurfacing after long inactivity
* niche-content consumers vs. mainstream viewers

Compared to KMeans, the HDBSCAN clusters displayed sharper boundaries, clearer identities, and higher internal consistency.

## **4.7 Overall Model Comparison**

The methods evaluated differ significantly in their assumptions and strengths:

| Method      | Strengths                                                        | Weaknesses                                                   | Outcome             |
| ----------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | ------------------- |
| **KMeans**  | Simple, stable, interpretable                                    | Assumes equal-density spherical clusters; forced assignments | Good baseline (k=7) |
| **GMM**     | Probabilistic assignments, flexible                              | Wrong assumptions for this dataset; clusters not meaningful  | Discarded           |
| **HDBSCAN** | Captures natural density; identifies noise; highly interpretable | Requires parameter tuning; leaves some users unclustered     | **Best model**      |

HDBSCAN was ultimately chosen because it offered the clearest segmentation, the best silhouette among meaningful models, and the most coherent behavioral grouping.

# **Final Conclusion**

Although KMeans provided a workable segmentation and GMM showed promising metrics without meaningful structure, **HDBSCAN emerged as the most powerful and insightful model**. It successfully captured the natural density structure of user behavior, created clear and interpretable groups, and avoided forcing irregular users into inappropriate clusters.

This makes HDBSCAN the recommended solution for understanding user behavior in this dataset.

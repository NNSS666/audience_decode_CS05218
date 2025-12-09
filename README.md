## Introduction
This project investigates viewer behaviour on a movie platform using the viewer_interactions.db database. This database includes several million user–movie ratings stored across multiple relational tables. The goal of this analysis was not to predict individual ratings, but to understand broader patterns in the audience engagement.

The README provides an overview of the project workflow, the rationale behind each design choice, and the analytical methods used to uncover audience behaviour from the underlying data.

## Methods

### Data Preparation
Transforming the raw SQLite database into a unified analytical dataset was an essential first step before any modelling could take place. The original database (viewer_interactions.db) consisted of five tables: viewer_ratings, movies, user_statistics, movie_statistics, and data_dictionary. Although the schema was structurally well-defined, the tables contained inconsistencies, missing values, and logical errors. Thus, the cleaning workflow focused on restoring data integrity, ensuring semantic consistency, and producing a dataset appropriate for machine learning.
#### ***1 Data Cleaning and Pre-Processing***
The viewer_ratings was the primary table for the entire project. Its preparation involved removing duplicated and anomalous rows, and correcting ratings outside the valid 1–5 range. These decisions were crucial because all later grouped calculations and downstream merges relied on this data frame.
The movies table required comparatively little intervention. Priority here was to ensure data integrity by removing duplicate movie_id entries, so that each movie had a unique reference record.
The user_statistics table required substantial reconstruction. Many fields, such as the number of activity days, mean rating, and standard deviation, were either missing or inconsistent. To correct this, the missing values were recomputed directly from the cleaned viewer_ratings table using grouped aggregates. Logical consistency rules were enforced, for example, ensuring that total_ratings always matched unique_movies, replacing invalid or contradictory values (such as impossible standard deviations), and converting floating-point columns to integers, where the data clearly represented integer values.
A similar process was applied to the movie_statistics table. First, it was verified that every movie referenced in this table actually appeared in viewer_ratings table. Later duplicates were removed, and missing statistics were recalculated from scratch. New feature (movie_rating_period) was also engineered which defined the number of days between the first and last rating a movie received. As with the user statistics, consistent imputation rules were applied to maintain logical coherence across the dataset.
Throughout the cleaning process, the data_dictionary table served as a reference to confirm column meanings, validate transformations, and maintain alignment with the intended semantics of each variable.
#### ***2 Logical Missing Value Imputation***
Because many dataset inconsistencies stemmed from missing or illogical statistical aggregates, structured imputation strategy was applied.
For users or movies with only a single rating,  the mean, min, and max were set equal to the rating itself. For all other missing cases, aggregates were recomputed, such as mean ratings or total counts, using groupby(customer_id) and groupby(movie_id) operations on the cleaned base table.
Standard deviation required special treatment. Following the logical constraint that dispersion cannot exist when only one observation is present, value was set to zero in those cases. Additionally,  where anomalies produced extreme deviation values, maximum cap of 2 was applied.
Any rating-related field that was outside the valid 1–5 range was corrected accordingly, and missing movie titles were labelled as “Unknown” to preserve dataset completeness during merging.
#### ***3 Final Merging and Dataset Construction***
Once all individual tables were cleaned, corrected, and logically standardized, they were merged into a consolidated dataset (initial_df). This merge relied on customer_id and movie_id as the joining keys, and naming conflicts were resolved using the suffixes _from_customer and _movie for clarity.
Also, a new categorical feature - release_time, was introduced to deal with cases where movie release years were missing. This categorical grouping helped preserve temporal information without creating misleading numerical values.
After merging, some statistical fields still contained missing values. Those values were recalculated using the same logic as with the separate statistics tables. For a few missing values, left after the recalculations, simple and consistent imputations were applied: minimum values were imputed with the minimum possible, maximum values with the maximum, averages with the mean, and dispersion fields using standard deviation. Any rows missing the rating variable were removed, since they cannot contribute to supervised model training.
This cleaning pipeline produced a coherent dataset that respected logical constraints, aligned with lesson principles on imputation and data validation, and set the foundation for exploratory analysis and machine learning.
### Exploratory Analysis
With a fully cleaned dataset, a quick exploratory analysis was done to understand feature behaviour and assess data suitability for machine learning models.
#### ***1 Correlation Analysis***
Using a correlation heatmap, relationships among all numerical variables were assessed. Strong correlations were found between logically related variables, such as: total_ratings and unique_movies_from_customer, or total_ratings_movie and unique_users_movie. To mitigate redundancy and collinearity issues, one variable from each pair was removed.
Multicollinearity was also widespread. Given that these statistics were derived from overlapping rating patterns, this result was expected. It further reinforced the importance of careful feature selection during the model training.
#### ***2 Distribution and Outlier Inspection***
Boxplots and distribution plots were generated to evaluate the spread of each feature. Many variables, such as user activity or movie rating periods, contained extreme outliers. Instead of removing these values outright, which could distort the real usage behaviour, this problem is addressed by using scaling techniques later on.
#### ***3 Pairwise Feature Relationships***
Finally, relationships between features were inspected. Different plots showed no strong or direct trends. This suggested that patterns in the dataset may be better explored through clustering techniques.
All these observations directly influenced modelling strategy used later in the project.

## Experimental Design 

### **Clustering Analysis**

#### **1 Feature Engineering and Data Preparation**

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

#### **2 Standardization and Dimensionality Reduction**

Because the engineered features operated on very different scales, the dataset was standardized using `StandardScaler`. This step ensured that models relying on distances or density estimation would not be biased by raw magnitudes.

To explore the structure of the data and assist visualization, PCA was applied.

* A **2-component PCA** captured ~47% of the total variance, enough to visualize broad patterns and potential cluster separations.
* A **5-component PCA** captured ~84% of the variance, providing a more stable low-dimensional space that density-based methods like HDBSCAN could use effectively.

PCA therefore played both an exploratory and a practical role in improving clustering performance.

#### **3 KMeans Clustering (k = 2–10)**

KMeans was used as a baseline model due to its simplicity and interpretability. We ran the algorithm for values of k ranging from 2 to 10, evaluating each configuration through inertia curves and silhouette scores.

While the inertia curve did not reveal a clear elbow point, the silhouette analysis provided more actionable insights. The strongest performers were:

* **k = 7** (silhouette ≈ 0.222)
* **k = 4** (silhouette ≈ 0.221)
* **k = 10** (silhouette ≈ 0.218)

Among these, **k = 7** offered the best compromise between behavioral detail and cluster separation. The resulting clusters were reasonably interpretable and relatively balanced in size, although some mixing of behaviors was still present due to the algorithm’s assumption of spherical and equally dense clusters.

#### **4 Gaussian Mixture Models (GMM)**

We then tested Gaussian Mixture Models, which provide soft assignments and can theoretically adapt to more flexible cluster shapes. Silhouette scores suggested that **k = 2** delivered the best result (≈ 0.245). However, this was misleading.

The underlying behavioral distributions are highly skewed, heavy-tailed, and vary greatly in density. These characteristics violate GMM’s Gaussian assumptions. As a result, although the model numerically achieved a good silhouette score, the resulting clusters were too broad and lacked behavioral meaning. They failed to differentiate between distinct patterns of engagement and rating behavior.

For this reason, despite promising initial metrics, **GMM was rejected**.

#### **5 HDBSCAN: Density-Based Clustering**

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

#### **6 Cluster Profiling and Behavioral Interpretation**

For all models, cluster profiles were analyzed through feature means and variances, enabling us to characterize each user segment.

HDBSCAN in particular produced clusters that aligned strongly with real behavioral patterns. Examples include:

* short-lived, harsh reviewers
* long-term users with consistent positivity
* moderately active users with stable habits
* dormant users resurfacing after long inactivity
* niche-content consumers vs. mainstream viewers

Compared to KMeans, the HDBSCAN clusters displayed sharper boundaries, clearer identities, and higher internal consistency.

#### **7 Overall Model Comparison**

The methods evaluated differ significantly in their assumptions and strengths:

| Method      | Strengths                                                        | Weaknesses                                                   | Outcome             |
| ----------- | ---------------------------------------------------------------- | ------------------------------------------------------------ | ------------------- |
| **KMeans**  | Simple, stable, interpretable                                    | Assumes equal-density spherical clusters; forced assignments | Good baseline (k=7) |
| **GMM**     | Probabilistic assignments, flexible                              | Wrong assumptions for this dataset; clusters not meaningful  | Discarded           |
| **HDBSCAN** | Captures natural density; identifies noise; highly interpretable | Requires parameter tuning; leaves some users unclustered     | **Best model**      |

HDBSCAN was ultimately chosen because it offered the clearest segmentation, the best silhouette among meaningful models, and the most coherent behavioral grouping.

#### **Final Conclusion**

Although KMeans provided a workable segmentation and GMM showed promising metrics without meaningful structure, **HDBSCAN emerged as the most powerful and insightful model**. It successfully captured the natural density structure of user behavior, created clear and interpretable groups, and avoided forcing irregular users into inappropriate clusters.

This makes HDBSCAN the recommended solution for understanding user behavior in this dataset.


## Results


## Conclusions
Our project showed that meaningful behavioral patterns can be identified even without predicting a specific target variable. The clustering analysis revealed clear user segments, ranging from short-lived harsh reviewers who abandoned the platform quickly (Cluster 0) to niche-focused critics (Cluster 1) and neutral users (Cluster 2). We also identified slightly longer-term light users who consistently rated poorly (Cluster 3) and uniformly positive legacy users (Cluster 4). The most valuable group consisted of long-term, recently active heavy users with diverse and informative rating behavior (Cluster 5). These findings highlight that engagement on the platform is structured and diverse, shaped by differences in activity duration, rating consistency, and content preference.

While our analysis captured clear engagement patterns, several questions remain open. The dataset lacks genre information, demographic details, and detailed viewing histories beyond rating timestamps, limiting our ability to understand why certain clusters behave as they do. Without genre or sequence data, we cannot analyze preference evolution or identify movie-specific causal factors. Future work could incorporate additional metadata and refine clustering by including richer features to better explain what differentiates long-term loyal users from short-lived disengaged ones.


# Databricks notebook source
# ============================================================
# NOTEBOOK 03: ML RANKING — COMPLETE (ALL-IN-ONE)
# ============================================================
import os
import mlflow
import mlflow.spark
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import (
    col, lit, when, length, rand, row_number, desc
)
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# ============================================================
# PART 1: LOAD DATA
# ============================================================
df_train_raw = spark.table("iitb.govscheme.eligibility_results")
print(f"✅ Loaded training data: {df_train_raw.count()} rows")

# ============================================================
# PART 2: CREATE RELEVANCE LABELS
# ============================================================
SEED = 123

df_labeled = df_train_raw \
    .withColumn("relevance_raw",
        (col("match_score") / 20.0)
        + (rand(SEED) * 1.5 - 0.75)
        + when(col("benefit_inr") >= 100000, 0.8)
         .when(col("benefit_inr") >= 50000, 0.4)
         .otherwise(0.0)
    ) \
    .withColumn("relevance",
        when(col("relevance_raw") >= 4.5, 5)
        .when(col("relevance_raw") >= 3.5, 4)
        .when(col("relevance_raw") >= 2.5, 3)
        .when(col("relevance_raw") >= 1.5, 2)
        .when(col("relevance_raw") >= 0.5, 1)
        .otherwise(0)
    ) \
    .drop("relevance_raw")

print("\n✅ Relevance distribution:")
df_labeled.groupBy("relevance").count().orderBy("relevance").show()

# ============================================================
# PART 3: FEATURE ENGINEERING
# ============================================================
occupation_indexer = StringIndexer(inputCol="occupation", outputCol="occupation_idx", handleInvalid="keep")
user_occupation_indexer = StringIndexer(inputCol="user_occupation", outputCol="user_occupation_idx", handleInvalid="keep")
caste_indexer = StringIndexer(inputCol="caste_eligible", outputCol="caste_idx", handleInvalid="keep")
gender_indexer = StringIndexer(inputCol="gender_eligible", outputCol="gender_idx", handleInvalid="keep")
state_indexer = StringIndexer(inputCol="state_eligible", outputCol="state_idx", handleInvalid="keep")

indexer_pipeline = Pipeline(stages=[
    occupation_indexer, user_occupation_indexer,
    caste_indexer, gender_indexer, state_indexer
])

indexer_model = indexer_pipeline.fit(df_labeled)
df_indexed = indexer_model.transform(df_labeled)

df_features = df_indexed \
    .withColumn("description_length", length(col("description"))) \
    .withColumn("benefit_log", F.log1p(col("benefit_inr"))) \
    .withColumn("income_gap", col("income_max_lpa") - col("user_income")) \
    .withColumn("age_range", col("age_max") - col("age_min")) \
    .withColumn("age_position",
        (col("user_age") - col("age_min")) / (col("age_max") - col("age_min") + 1)
    ) \
    .withColumn("occupation_match",
        when(col("occupation") == col("user_occupation"), 1.0)
        .when(col("occupation") == "any", 0.5)
        .otherwise(0.0)
    ) \
    .withColumn("gender_match",
        when(col("gender_eligible") == col("user_gender"), 1.0)
        .when(col("gender_eligible") == "all", 0.5)
        .otherwise(0.0)
    ) \
    .withColumn("rural_match",
        when(col("is_rural") == col("user_is_rural"), 1.0).otherwise(0.0)
    )

feature_columns = [
    "benefit_inr", "benefit_log", "income_max_lpa", "income_gap",
    "age_min", "age_max", "age_range", "age_position",
    "occupation_idx", "user_occupation_idx", "caste_idx",
    "gender_idx", "state_idx",
    "occupation_match", "gender_match", "rural_match",
    "is_rural", "user_is_rural", "user_age", "user_income",
    "description_length", "match_score"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="keep"
)

df_ml = assembler.transform(df_features)
print(f"✅ Features ready: {df_ml.count()} rows, {len(feature_columns)} features")

# ============================================================
# PART 4: TRAIN-TEST SPLIT
# ============================================================
df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)
print(f"✅ Train: {df_train.count()} | Test: {df_test.count()}")

# ============================================================
# PART 5: TRAIN MODEL + MLFLOW
# ============================================================
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/iitb/govscheme/raw_data/mlflow_tmp"
mlflow.autolog(disable=True)

with mlflow.start_run(run_name="govscheme_ranker_v1") as run:

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="relevance",
        maxDepth=5,
        maxIter=50,
        stepSize=0.1,
        seed=42
    )

    model = gbt.fit(df_train)
    predictions = model.transform(df_test)

    evaluator_rmse = RegressionEvaluator(labelCol="relevance", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="relevance", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    mlflow.log_param("model_type", "GBTRegressor")
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("max_iter", 50)
    mlflow.log_param("num_features", len(feature_columns))
    mlflow.log_param("train_size", df_train.count())
    mlflow.log_param("test_size", df_test.count())
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.spark.log_model(
        model,
        "govscheme_ranker",
        dfs_tmpdir="/Volumes/iitb/govscheme/raw_data/mlflow_tmp"
    )

    importances = model.featureImportances.toArray()
    for fname, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        mlflow.log_metric(f"fi_{fname}", round(imp, 4))

    run_id = run.info.run_id

print(f"""
╔══════════════════════════════════════════╗
║   ML MODEL TRAINING COMPLETE            ║
╠══════════════════════════════════════════╣
║  Model:     GBT Regressor (Ranking)     ║
║  RMSE:      {rmse:.4f}                       ║
║  R²:        {r2:.4f}                       ║
║  MLflow Run: {run_id[:20]}...  ║
╚══════════════════════════════════════════╝
""")

print("Sample predictions:")
predictions.select(
    "schemeName", "relevance", "prediction", "benefit_inr", "match_score"
).orderBy(col("prediction").desc()).show(10, truncate=45)

# ============================================================
# PART 6: FEATURE IMPORTANCE
# ============================================================
import pandas as pd

fi_data = list(zip(feature_columns, model.featureImportances.toArray()))
fi_df = pd.DataFrame(fi_data, columns=["feature", "importance"]).sort_values("importance", ascending=False)

print("\n🏆 TOP FEATURES FOR SCHEME RANKING:")
print("=" * 50)
for _, row in fi_df.head(10).iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:25s} {row['importance']:.4f}  {bar}")

df_fi = spark.createDataFrame(fi_df)
df_fi.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("iitb.govscheme.feature_importance")
print(f"\n✅ Feature importance saved to Delta")

# COMMAND ----------

# CELL 2: rank_schemes function + get_eligible_schemes (SELF-CONTAINED)
from pyspark.sql.functions import (
    col, lit, when, length, lower, desc, row_number
)
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def get_eligible_schemes(user_profile):
    df = spark.table("iitb.govscheme.schemes")
    age = user_profile.get("age", 25)
    income = user_profile.get("income_lpa", 3.0)
    occupation = user_profile.get("occupation", "any").lower().strip()
    caste = user_profile.get("caste", "general").lower().strip()
    gender = user_profile.get("gender", "all").lower().strip()
    is_rural = user_profile.get("is_rural", 0)
    state = user_profile.get("state", "all").lower().strip()

    df_eligible = df.filter(
        (col("age_min") <= age) & (col("age_max") >= age)
    ).filter(
        col("income_max_lpa") >= income
    ).filter(
        (lower(col("occupation")) == occupation) | (lower(col("occupation")) == "any")
    ).filter(
        (lower(col("gender_eligible")) == gender) | (lower(col("gender_eligible")) == "all")
    ).filter(
        (lower(col("caste_eligible")).contains(caste)) | (lower(col("caste_eligible")) == "all")
    ).withColumn("match_score",
        lit(0)
        + when(lower(col("occupation")) == occupation, 20).otherwise(0)
        + when(lower(col("caste_eligible")).contains(caste), 15).otherwise(0)
        + when((col("is_rural") == 1) & (lit(is_rural) == 1), 10).otherwise(0)
        + when((lower(col("state_eligible")) == state) | (lower(col("state_eligible")) == "all"), 15).otherwise(0)
        + when(col("benefit_inr") >= 100000, 20)
         .when(col("benefit_inr") >= 50000, 15)
         .when(col("benefit_inr") >= 25000, 10)
         .when(col("benefit_inr") >= 10000, 5)
         .otherwise(0)
    ).orderBy(desc("match_score"), desc("benefit_inr"))

    return df_eligible


def rank_schemes(df_eligible, user_profile):
    df = df_eligible \
        .withColumn("user_age", lit(user_profile.get("age", 30))) \
        .withColumn("user_income", lit(user_profile.get("income_lpa", 3.0))) \
        .withColumn("user_occupation", lit(user_profile.get("occupation", "any"))) \
        .withColumn("user_gender", lit(user_profile.get("gender", "male"))) \
        .withColumn("user_is_rural", lit(user_profile.get("is_rural", 0))) \
        .withColumn("user_state", lit(user_profile.get("state", "all")))

    df = indexer_model.transform(df)

    df = df \
        .withColumn("description_length", length(col("description"))) \
        .withColumn("benefit_log", F.log1p(col("benefit_inr"))) \
        .withColumn("income_gap", col("income_max_lpa") - col("user_income")) \
        .withColumn("age_range", col("age_max") - col("age_min")) \
        .withColumn("age_position",
            (col("user_age") - col("age_min")) / (col("age_max") - col("age_min") + 1)
        ) \
        .withColumn("occupation_match",
            when(col("occupation") == col("user_occupation"), 1.0)
            .when(col("occupation") == "any", 0.5)
            .otherwise(0.0)
        ) \
        .withColumn("gender_match",
            when(col("gender_eligible") == col("user_gender"), 1.0)
            .when(col("gender_eligible") == "all", 0.5)
            .otherwise(0.0)
        ) \
        .withColumn("rural_match",
            when(col("is_rural") == col("user_is_rural"), 1.0).otherwise(0.0)
        )

    df = assembler.transform(df)
    df_ranked = model.transform(df)

    window = Window.orderBy(desc("prediction"))
    df_ranked = df_ranked.withColumn("ml_rank", row_number().over(window))

    result = df_ranked.select(
        "ml_rank",
        "schemeName",
        "schemeId",
        "schemeSlug",
        "category_str",
        "ministry",
        "description",
        "benefits_text",
        "eligibility_text",
        "application_process",
        "benefit_inr",
        "match_score",
        col("prediction").alias("ml_score"),
    ).orderBy("ml_rank")

    return result

print("✅ get_eligible_schemes() defined")
print("✅ rank_schemes() defined")

# COMMAND ----------

# CELL 3: Test + Save ranked results

test_profile = {
    "age": 25,
    "income_lpa": 2.5,
    "occupation": "farmer",
    "caste": "obc",
    "gender": "male",
    "is_rural": 1,
    "state": "maharashtra"
}

# Step 1: Eligibility
df_eligible = get_eligible_schemes(test_profile)
elig_count = df_eligible.count()
print(f"Step 1 — Eligible: {elig_count}")

# Step 2: ML Ranking
df_ranked = rank_schemes(df_eligible, test_profile)
rank_count = df_ranked.count()
print(f"Step 2 — Ranked: {rank_count}")

print("\n🏆 TOP 15 ML-RANKED SCHEMES:")
df_ranked.select(
    "ml_rank", "schemeName", "benefit_inr", "ml_score", "match_score", "category_str"
).show(15, truncate=50)

# Save to Delta
df_ranked.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("iitb.govscheme.ranked_results")

# Verify
verify_count = spark.table("iitb.govscheme.ranked_results").count()
print(f"✅ Ranked results saved: {verify_count} rows")

print("""
╔══════════════════════════════════════════════════╗
║   NOTEBOOK 03: COMPLETE ✅                       ║
╠══════════════════════════════════════════════════╣
║  Ready for Notebook 04 (Optimization)            ║
║  Ready for Notebook 05 (Pipeline Integration)    ║
╚══════════════════════════════════════════════════╝
""")

# COMMAND ----------

# Quick verify
print(spark.table("iitb.govscheme.ranked_results").count())

# COMMAND ----------


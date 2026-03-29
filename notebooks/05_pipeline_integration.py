# Databricks notebook source
# MAGIC %md
# MAGIC # GovScheme-AI: End-to-End Pipeline Integration
# MAGIC # This notebook integrates eligibility filtering, ranking, and optimization

# COMMAND ----------

# =============================================================
# NOTEBOOK 05: GovScheme-AI — Full Pipeline Integration & Test
# =============================================================
# Self-contained: reloads all data, retrains model, tests 3 users
# Databricks Serverless compatible — NO RDDs, NO DBFS
# =============================================================

import os
import re
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Spark + Delta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# ML
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

# MLflow
import mlflow
import mlflow.spark
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/iitb/govscheme/raw_data/mlflow_tmp"

warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("=" * 70)
print("   GovScheme-AI: Full Pipeline Integration")
print("   Notebook 05 — Self-contained reload + test")
print("=" * 70)

# COMMAND ----------

# =============================================================
# STEP 1: Load schemes from Delta
# =============================================================
df_schemes = spark.read.table("iitb.govscheme.schemes")
SCHEMES_COUNT = df_schemes.count()
print(f"✅ Loaded schemes table: {SCHEMES_COUNT} rows")
df_schemes.printSchema()
df_schemes.show(3, truncate=60)

# COMMAND ----------

# =============================================================
# STEP 2: User Profile Parser (NLP — regex-based)
# =============================================================

def parse_user_profile(text):
    """
    Parse natural language user description into structured profile dict.
    Extracts: age, income, occupation, gender, caste, rural/urban, state.
    """
    text_lower = text.lower().strip()
    profile = {
        "age": 30,
        "income_lpa": 3.0,
        "occupation": "general",
        "gender": "any",
        "caste": "general",
        "is_rural": 0,
        "state": "all",
        "description": text
    }

    # --- Age extraction ---
    age_patterns = [
        r'(\d{1,3})\s*(?:years?\s*old|yr|yrs|age)',
        r'age\s*(?:is|:)?\s*(\d{1,3})',
        r'i\s+am\s+(\d{1,3})',
        r'(\d{2})\s*(?:male|female|m|f)\b'
    ]
    for pat in age_patterns:
        m = re.search(pat, text_lower)
        if m:
            age_val = int(m.group(1))
            if 1 <= age_val <= 120:
                profile["age"] = age_val
                break

    # --- Income extraction ---
    income_patterns = [
        (r'(?:income|earn(?:ing)?|salary|stipend)\s*(?:is|of|:)?\s*(?:rs\.?|₹|inr)?\s*([\d,.]+)\s*(?:lpa|lac|lakh|per\s*annum)', "lpa"),
        (r'(?:rs\.?|₹|inr)\s*([\d,.]+)\s*(?:lpa|lac|lakh)', "lpa"),
        (r'([\d,.]+)\s*(?:lpa|lac\s*per\s*annum)', "lpa"),
        (r'(?:income|earn(?:ing)?|salary)\s*(?:is|of|:)?\s*(?:rs\.?|₹|inr)?\s*([\d,.]+)\s*(?:per\s*month|monthly|pm)', "monthly"),
        (r'(?:income|earn(?:ing)?|salary)\s*(?:is|of|:)?\s*(?:rs\.?|₹|inr)?\s*([\d,.]+)\s*(?:per\s*year|yearly|annual)', "yearly"),
        (r'(?:rs\.?|₹|inr)\s*([\d,.]+)\s*(?:per\s*month|monthly|pm)', "monthly"),
        (r'(?:rs\.?|₹|inr)\s*([\d,.]+)\s*(?:per\s*year|yearly|annual)', "yearly"),
    ]
    for pat, mode in income_patterns:
        m = re.search(pat, text_lower)
        if m:
            val_str = m.group(1).replace(",", "")
            val = float(val_str)
            if mode == "monthly":
                profile["income_lpa"] = round((val * 12) / 100000, 2)
            elif mode == "yearly":
                profile["income_lpa"] = round(val / 100000, 2)
            else:
                profile["income_lpa"] = val
            break

    # --- Occupation ---
    occupation_map = {
        "farmer": ["farmer", "agriculture", "farming", "kisan", "krishi", "cultivat"],
        "student": ["student", "college", "university", "school", "study", "pursuing",
                     "undergraduate", "graduate", "phd", "engineering", "btech", "mtech"],
        "women": ["woman", "women", "female", "girl", "housewife", "mother", "widow",
                   "pregnant", "mahila"],
        "senior_citizen": ["senior", "elderly", "retired", "pension", "old age",
                           "60 year", "65 year", "70 year", "75 year"],
        "worker": ["worker", "labour", "labor", "daily wage", "construction",
                    "factory", "maid", "domestic"],
        "entrepreneur": ["entrepreneur", "business", "startup", "self-employed",
                         "self employed", "shop", "enterprise", "msme", "udyam"],
        "disabled": ["disabled", "disability", "handicap", "divyang", "blind",
                      "deaf", "impair"],
        "minority": ["minority", "muslim", "christian", "sikh", "buddhist",
                      "jain", "parsi"],
    }
    for occ, keywords in occupation_map.items():
        if any(kw in text_lower for kw in keywords):
            profile["occupation"] = occ
            break

    # --- Gender ---
    if any(w in text_lower for w in ["female", "woman", "girl", "she ", "her ",
                                      "mother", "wife", "widow", "mahila"]):
        profile["gender"] = "female"
    elif any(w in text_lower for w in ["male", "man", "boy", "he ", "his ",
                                        "father", "husband"]):
        profile["gender"] = "male"

    # --- Caste ---
    for caste in ["sc", "st", "obc", "general", "ews"]:
        if re.search(r'\b' + caste + r'\b', text_lower):
            profile["caste"] = caste
            break

    # --- Rural/Urban ---
    if any(w in text_lower for w in ["rural", "village", "gram", "gaon", "panchayat"]):
        profile["is_rural"] = 1
    elif any(w in text_lower for w in ["urban", "city", "metro", "town", "municipal"]):
        profile["is_rural"] = 0

    # --- State ---
    states = [
        "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
        "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
        "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
        "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
        "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
        "delhi", "jammu", "kashmir", "ladakh"
    ]
    for st in states:
        if st in text_lower:
            profile["state"] = st
            break

    return profile

print("✅ parse_user_profile() defined")

# COMMAND ----------

# =============================================================
# STEP 3: Eligibility Filtering Engine
# =============================================================

def get_eligible_schemes(user_profile):
    """
    Filter schemes based on user profile.
    Returns Spark DataFrame with match_score column.
    """
    df = df_schemes

    age = user_profile.get("age", 30)
    income = user_profile.get("income_lpa", 3.0)
    occupation = user_profile.get("occupation", "general")
    gender = user_profile.get("gender", "any")
    caste = user_profile.get("caste", "general")
    is_rural = user_profile.get("is_rural", 0)
    state = user_profile.get("state", "all")

    # Core eligibility filters
    df_eligible = df.filter(
        (F.col("age_min") <= age) &
        (F.col("age_max") >= age) &
        (F.col("income_max_lpa") >= income)
    )

    # Gender filter
    if gender != "any":
        df_eligible = df_eligible.filter(
            (F.col("gender_eligible") == gender) |
            (F.col("gender_eligible") == "all") |
            (F.col("gender_eligible").isNull())
        )

    # Caste filter
    if caste != "general":
        df_eligible = df_eligible.filter(
            F.col("caste_eligible").contains(caste) |
            F.col("caste_eligible").contains("all") |
            F.col("caste_eligible").isNull()
        )

    # Compute match score
    df_scored = df_eligible.withColumn(
        "match_score",
        F.lit(20) +
        F.when(F.lower(F.col("occupation")) == F.lit(occupation), F.lit(20)).otherwise(F.lit(0)) +
        F.when(F.col("caste_eligible").contains(caste), F.lit(15)).otherwise(F.lit(0)) +
        F.when(F.col("is_rural") == is_rural, F.lit(10)).otherwise(F.lit(0)) +
        F.when(
            (F.col("state_eligible") == "all") | F.col("state_eligible").contains(state),
            F.lit(15)
        ).otherwise(F.lit(0)) +
        F.when(F.col("benefit_inr") >= 500000, F.lit(20))
         .when(F.col("benefit_inr") >= 100000, F.lit(15))
         .when(F.col("benefit_inr") >= 10000, F.lit(10))
         .otherwise(F.lit(5))
    )

    return df_scored

print("✅ get_eligible_schemes() defined")

# COMMAND ----------

# =============================================================
# STEP 4: Retrain ML Ranker (required — variables don't persist)
# =============================================================
print("Training ML ranker from iitb.govscheme.eligibility_results...")

df_train = spark.read.table("iitb.govscheme.eligibility_results")
train_count = df_train.count()
print(f"   Loaded {train_count} training rows")

# --- Relevance label ---
df_train = df_train.withColumn(
    "relevance",
    F.least(
        F.lit(5),
        F.greatest(
            F.lit(0),
            F.round(
                (F.col("match_score") / 20.0) +
                F.when(F.col("benefit_inr") >= 500000, F.lit(1.5))
                 .when(F.col("benefit_inr") >= 100000, F.lit(1.0))
                 .when(F.col("benefit_inr") >= 10000, F.lit(0.5))
                 .otherwise(F.lit(0.0)) +
                F.lit(0.5) * (F.rand() - 0.5),
                1
            )
        )
    )
)

# --- Feature engineering ---
df_train = df_train.withColumn("benefit_log", F.log1p(F.col("benefit_inr")))
df_train = df_train.withColumn("age_range", F.col("age_max") - F.col("age_min"))
df_train = df_train.withColumn("description_length",
    F.when(F.col("description").isNotNull(), F.length(F.col("description"))).otherwise(F.lit(0))
)
df_train = df_train.withColumn("income_gap", F.col("income_max_lpa") - F.col("user_income"))
df_train = df_train.withColumn("age_position",
    F.when(F.col("age_range") > 0,
        (F.col("user_age") - F.col("age_min")) / F.col("age_range")
    ).otherwise(F.lit(0.5))
)
df_train = df_train.withColumn("occupation_match",
    F.when(F.lower(F.col("occupation")) == F.lower(F.col("user_occupation")), F.lit(1)).otherwise(F.lit(0))
)
df_train = df_train.withColumn("gender_match",
    F.when(
        (F.col("gender_eligible") == "all") |
        (F.lower(F.col("gender_eligible")) == F.lower(F.col("user_gender"))),
        F.lit(1)
    ).otherwise(F.lit(0))
)
df_train = df_train.withColumn("rural_match",
    F.when(F.col("is_rural") == F.col("user_is_rural"), F.lit(1)).otherwise(F.lit(0))
)

# --- Fill nulls in categoricals ---
categorical_cols = ["occupation", "caste_eligible", "gender_eligible", "state_eligible", "user_occupation"]
for col_name in categorical_cols:
    df_train = df_train.withColumn(
        col_name,
        F.when(F.col(col_name).isNull(), "unknown").otherwise(F.col(col_name))
    )

# --- StringIndexers ---
indexers = []
indexed_cols = []
for col_name in categorical_cols:
    out_col = col_name + "_idx"
    indexed_cols.append(out_col)
    indexers.append(StringIndexer(inputCol=col_name, outputCol=out_col, handleInvalid="keep"))

# --- Feature list ---
feature_cols = [
    "benefit_inr", "benefit_log", "income_max_lpa", "age_min", "age_max", "age_range",
    "description_length", "user_age", "user_income", "user_is_rural",
    "income_gap", "age_position", "occupation_match", "gender_match", "rural_match",
    "match_score"
] + indexed_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="relevance",
    predictionCol="ml_score",
    maxDepth=5,
    maxIter=50,
    stepSize=0.1
)

pipeline = Pipeline(stages=indexers + [assembler, gbt])

# --- Train with MLflow logging ---
# FIXED: hardcoded experiment name (spark.databricks.notebook.path not available on Serverless)
mlflow.set_experiment("/govscheme_pipeline_v2")

try:
    with mlflow.start_run(run_name="NB05_pipeline_retrain") as run:
        model = pipeline.fit(df_train)

        # Evaluate
        predictions = model.transform(df_train)
        from pyspark.ml.evaluation import RegressionEvaluator
        evaluator_rmse = RegressionEvaluator(labelCol="relevance", predictionCol="ml_score", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol="relevance", predictionCol="ml_score", metricName="r2")
        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        mlflow.log_param("maxDepth", 5)
        mlflow.log_param("maxIter", 50)
        mlflow.log_param("stepSize", 0.1)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("training_rows", train_count)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print(f"✅ ML Model trained — RMSE={rmse:.4f}, R²={r2:.4f}")
        print(f"   MLflow run_id: {run.info.run_id}")

except Exception as e:
    print(f"⚠️ MLflow logging failed (non-fatal): {e}")
    print("   Training without MLflow...")
    model = pipeline.fit(df_train)
    predictions = model.transform(df_train)
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator_rmse = RegressionEvaluator(labelCol="relevance", predictionCol="ml_score", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="relevance", predictionCol="ml_score", metricName="r2")
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    print(f"✅ ML Model trained (no MLflow) — RMSE={rmse:.4f}, R²={r2:.4f}")

print(f"✅ Model ready for inference")

# COMMAND ----------

# =============================================================
# STEP 5: Ranking Function
# =============================================================

def rank_schemes(df_eligible, user_profile):
    """Apply trained ML model to score and rank eligible schemes."""

    age = user_profile.get("age", 30)
    income = user_profile.get("income_lpa", 3.0)
    occupation = user_profile.get("occupation", "general")
    gender = user_profile.get("gender", "any")
    is_rural = user_profile.get("is_rural", 0)

    # Add user feature columns
    df = df_eligible \
        .withColumn("user_age", F.lit(age)) \
        .withColumn("user_income", F.lit(income)) \
        .withColumn("user_occupation", F.lit(occupation)) \
        .withColumn("user_gender", F.lit(gender)) \
        .withColumn("user_is_rural", F.lit(is_rural))

    # Feature engineering — must match training
    df = df.withColumn("benefit_log", F.log1p(F.col("benefit_inr")))
    df = df.withColumn("age_range", F.col("age_max") - F.col("age_min"))
    df = df.withColumn("description_length",
        F.when(F.col("description").isNotNull(), F.length(F.col("description"))).otherwise(F.lit(0))
    )
    df = df.withColumn("income_gap", F.col("income_max_lpa") - F.col("user_income"))
    df = df.withColumn("age_position",
        F.when(F.col("age_range") > 0,
            (F.col("user_age") - F.col("age_min")) / F.col("age_range")
        ).otherwise(F.lit(0.5))
    )
    df = df.withColumn("occupation_match",
        F.when(F.lower(F.col("occupation")) == F.lower(F.col("user_occupation")), F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("gender_match",
        F.when(
            (F.col("gender_eligible") == "all") |
            (F.lower(F.col("gender_eligible")) == F.lower(F.col("user_gender"))),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    df = df.withColumn("rural_match",
        F.when(F.col("is_rural") == F.col("user_is_rural"), F.lit(1)).otherwise(F.lit(0))
    )

    # Fill nulls in categoricals
    for col_name in categorical_cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), "unknown").otherwise(F.col(col_name))
        )

    # Apply model
    df_scored = model.transform(df)

    # Clamp ml_score to [0, 5]
    df_scored = df_scored.withColumn("ml_score",
        F.least(F.lit(5.0), F.greatest(F.lit(0.0), F.col("ml_score")))
    )

    return df_scored.orderBy(F.col("ml_score").desc())

print("✅ rank_schemes() defined")

# COMMAND ----------

# =============================================================
# STEP 6: Bundle Optimizer
# =============================================================

def optimize_bundle(df_ranked, max_schemes=10, max_per_category=3):
    """
    Greedy diverse selection: cap per category, take top N globally.
    Returns Pandas DataFrame.
    """
    pdf = df_ranked.select(
        "schemeId", "schemeName", "category_str", "benefit_inr",
        "ml_score", "match_score", "description", "ministry",
        "eligibility_text", "benefits_text", "application_process",
        "occupation", "gender_eligible", "caste_eligible", "state_eligible",
        "age_min", "age_max", "income_max_lpa"
    ).toPandas()

    if pdf.empty:
        return pdf

    pdf = pdf.sort_values("ml_score", ascending=False).reset_index(drop=True)

    selected = []
    cat_count = {}

    for _, row in pdf.iterrows():
        cat = str(row.get("category_str", "Other"))
        if cat_count.get(cat, 0) < max_per_category:
            selected.append(row)
            cat_count[cat] = cat_count.get(cat, 0) + 1
            if len(selected) >= max_schemes:
                break

    result = pd.DataFrame(selected)
    return result


def get_bundle_summary(bundle_df):
    """Generate summary stats for a bundle."""
    if bundle_df.empty:
        return {"error": "No schemes in bundle"}

    return {
        "total_schemes": len(bundle_df),
        "total_benefit_inr": int(bundle_df["benefit_inr"].sum()),
        "total_benefit_str": f"₹{bundle_df['benefit_inr'].sum():,.0f}",
        "categories_covered": int(bundle_df["category_str"].nunique()),
        "avg_ml_score": round(float(bundle_df["ml_score"].mean()), 3),
        "max_ml_score": round(float(bundle_df["ml_score"].max()), 3),
        "min_ml_score": round(float(bundle_df["ml_score"].min()), 3),
    }

print("✅ optimize_bundle() and get_bundle_summary() defined")

# COMMAND ----------

# =============================================================
# STEP 7: Master Pipeline — govscheme_ai()
# =============================================================

def govscheme_ai(user_input, max_schemes=10, verbose=True):
    """
    End-to-end pipeline:
    text → parse → filter → rank → optimize → results
    
    Returns: (profile_dict, bundle_pandas_df, summary_dict)
    """
    if verbose:
        print("=" * 70)
        print(f"🔍 INPUT: {user_input[:100]}...")
        print("=" * 70)

    # Step 1: Parse
    profile = parse_user_profile(user_input)
    if verbose:
        print(f"\n👤 PARSED PROFILE:")
        for k, v in profile.items():
            if k != "description":
                print(f"   {k:15s}: {v}")

    # Step 2: Filter
    df_eligible = get_eligible_schemes(profile)
    eligible_count = df_eligible.count()
    if verbose:
        print(f"\n📋 ELIGIBLE SCHEMES: {eligible_count}")

    if eligible_count == 0:
        print("   ⚠️ No eligible schemes found for this profile!")
        return profile, pd.DataFrame(), {"error": "No eligible schemes found"}

    # Step 3: Rank
    df_ranked = rank_schemes(df_eligible, profile)
    if verbose:
        print(f"   ✅ Ranked {eligible_count} schemes with ML model")

    # Step 4: Optimize
    bundle = optimize_bundle(df_ranked, max_schemes=max_schemes)
    summary = get_bundle_summary(bundle)

    if verbose:
        print(f"\n🏆 OPTIMIZED BUNDLE:")
        print(f"   Schemes selected : {summary.get('total_schemes', 0)}")
        print(f"   Total benefit    : {summary.get('total_benefit_str', '₹0')}")
        print(f"   Categories       : {summary.get('categories_covered', 0)}")
        print(f"   Avg ML score     : {summary.get('avg_ml_score', 0)}/5.0")
        print(f"   Score range      : {summary.get('min_ml_score', 0)} — {summary.get('max_ml_score', 0)}")

        print(f"\n📄 TOP SCHEMES:")
        print("-" * 70)
        for i, (_, row) in enumerate(bundle.iterrows(), 1):
            score_bar = "█" * int(row["ml_score"]) + "░" * (5 - int(row["ml_score"]))
            print(f"   {i:2d}. [{score_bar}] {row['ml_score']:.2f}  ₹{row['benefit_inr']:>10,.0f}  {row['schemeName'][:55]}")
            print(f"       Ministry: {str(row.get('ministry', 'N/A'))[:50]}")
            print(f"       Category: {row.get('category_str', 'N/A')}")
            print()

    return profile, bundle, summary

print("✅ govscheme_ai() master pipeline defined")
print("=" * 70)
print("🚀 ALL FUNCTIONS READY — Running tests below...")
print("=" * 70)

# COMMAND ----------

# =============================================================
# TEST 1: Young Female SC Student from Rural Rajasthan
# =============================================================

test1_input = """
I am a 22 year old female student from a rural village in Rajasthan. 
My family income is 2 lakh per annum. I belong to SC category and 
I am currently pursuing engineering at a government college.
"""

profile1, bundle1, summary1 = govscheme_ai(test1_input, max_schemes=10)
print(f"\n✅ TEST 1 RESULT: {summary1}")

# COMMAND ----------

# =============================================================
# TEST 2: Middle-aged Male OBC Farmer from Rural Maharashtra
# =============================================================

test2_input = """
I am a 45 year old male farmer from a village in Maharashtra. 
My annual income is 1.5 lakh. I belong to OBC category. I own 
2 acres of agricultural land and grow rice and wheat.
"""

profile2, bundle2, summary2 = govscheme_ai(test2_input, max_schemes=10)
print(f"\n✅ TEST 2 RESULT: {summary2}")

# COMMAND ----------

# =============================================================
# TEST 3: Senior Citizen Retired Male from Delhi
# =============================================================

test3_input = """
I am a 68 year old retired senior citizen living in Delhi. 
My pension income is 3 lakh per annum. I am a general category male.
I need schemes related to healthcare and old age support.
"""

profile3, bundle3, summary3 = govscheme_ai(test3_input, max_schemes=8)
print(f"\n✅ TEST 3 RESULT: {summary3}")

# COMMAND ----------

# =============================================================
# FINAL SUMMARY: All 3 Test Users
# =============================================================

print("=" * 70)
print("   📊 GOVSCHEME-AI: PIPELINE VERIFICATION SUMMARY")
print("=" * 70)

test_results = [
    ("Test 1: 22F Student SC Rural Rajasthan", summary1),
    ("Test 2: 45M Farmer OBC Rural Maharashtra", summary2),
    ("Test 3: 68M Senior General Delhi", summary3),
]

all_passed = True
for name, summary in test_results:
    status = "✅ PASS" if summary.get("total_schemes", 0) > 0 else "❌ FAIL"
    if summary.get("total_schemes", 0) == 0:
        all_passed = False
    print(f"\n{status} — {name}")
    for k, v in summary.items():
        print(f"   {k:25s}: {v}")

print("\n" + "=" * 70)
if all_passed:
    print("🎉 ALL TESTS PASSED — Pipeline is fully functional!")
    print("   → Proceed to Notebook 06 (Gradio UI)")
else:
    print("⚠️ SOME TESTS FAILED — check eligibility filters")
print("=" * 70)

# --- Save one sample bundle to Delta for reference ---
try:
    bundle_spark = spark.createDataFrame(bundle1)
    bundle_spark.write.mode("overwrite").saveAsTable("iitb.govscheme.pipeline_test_bundle")
    print("\n✅ Sample bundle saved to iitb.govscheme.pipeline_test_bundle")
except Exception as e:
    print(f"\n⚠️ Could not save test bundle: {e}")

# COMMAND ----------


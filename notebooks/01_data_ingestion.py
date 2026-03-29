# Databricks notebook source
# Cell 1: Verify setup
import mlflow
print(f"Spark version: {spark.version}")
print(f"MLflow version: {mlflow.__version__}")
spark.sql("USE CATALOG iitb")
spark.sql("USE SCHEMA govscheme")
print("✓ Catalog and schema set")

# COMMAND ----------

# Cell 2: Create schemes table (clean slate)
spark.sql("""
CREATE OR REPLACE TABLE iitb.govscheme.schemes (
    scheme_id        INT     NOT NULL,
    name             STRING  NOT NULL,
    ministry         STRING,
    category         STRING  NOT NULL,
    state            STRING  NOT NULL,
    gender           STRING,
    age_min          INT,
    age_max          INT,
    income_max_lpa   DOUBLE,
    occupation       STRING,
    caste_eligible   STRING,
    benefit_inr      BIGINT,
    benefit_type     STRING,
    application_url  STRING,
    description      STRING
)
USING DELTA
PARTITIONED BY (category, state)
COMMENT 'Government schemes master table - GovScheme AI'
""")
print("✓ schemes table ready")

# COMMAND ----------

# Cell 3: Create profiles table (clean slate)
spark.sql("""
CREATE OR REPLACE TABLE iitb.govscheme.profiles (
    profile_id        STRING  NOT NULL,
    age               INT     NOT NULL,
    gender            STRING  NOT NULL,
    state             STRING  NOT NULL,
    income_lpa        DOUBLE  NOT NULL,
    occupation        STRING  NOT NULL,
    caste             STRING,
    has_bpl_card      STRING,
    land_owned_acres  DOUBLE
)
USING DELTA
PARTITIONED BY (state)
COMMENT 'User profiles table - GovScheme AI'
""")
print("✓ profiles table ready")

# COMMAND ----------

# Cell 4: Generate and write 500 synthetic user profiles
import random
import uuid

random.seed(42)

states      = ["Rajasthan", "Maharashtra", "Bihar", "Uttar Pradesh",
               "Tamil Nadu", "Karnataka", "West Bengal", "Madhya Pradesh",
               "Gujarat", "Odisha"]
occupations = ["Farmer", "Student", "Daily Wage Worker",
               "Small Business", "Government Employee", "Unemployed"]
castes      = ["General", "OBC", "SC", "ST"]
genders     = ["Male", "Female"]

profiles = []
for _ in range(500):
    occ = random.choice(occupations)
    if occ == "Farmer":
        income = round(random.uniform(0.5, 3.0), 2)
    elif occ == "Student":
        income = round(random.uniform(0.0, 1.0), 2)
    elif occ == "Government Employee":
        income = round(random.uniform(3.0, 12.0), 2)
    else:
        income = round(random.uniform(0.5, 5.0), 2)

    land = round(random.uniform(0.5, 5.0), 1) if occ == "Farmer" else 0.0

    profiles.append((
        str(uuid.uuid4()),          # profile_id
        random.randint(18, 70),     # age
        random.choice(genders),     # gender
        random.choice(states),      # state
        float(income),              # income_lpa
        occ,                        # occupation
        random.choice(castes),      # caste
        random.choice(["Yes","No"]),# has_bpl_card
        float(land),                # land_owned_acres
    ))

from pyspark.sql.types import *
profiles_schema = StructType([
    StructField("profile_id",       StringType(), False),
    StructField("age",              IntegerType(), False),
    StructField("gender",           StringType(), False),
    StructField("state",            StringType(), False),
    StructField("income_lpa",       DoubleType(),  False),
    StructField("occupation",       StringType(), False),
    StructField("caste",            StringType(), True),
    StructField("has_bpl_card",     StringType(), True),
    StructField("land_owned_acres", DoubleType(),  True),
])

profiles_df = spark.createDataFrame(profiles, schema=profiles_schema)

profiles_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("iitb.govscheme.profiles")

print(f"✓ {profiles_df.count()} profiles written to iitb.govscheme.profiles")
profiles_df.groupBy("occupation").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# Cell 5: Load real schemes data from Person B
# ---------------------------------------------------
# When Person B sends the CSV, upload it via:
# Catalog → + Add data → Upload file
# Then replace the path below with the actual volume path
# ---------------------------------------------------

# PLACEHOLDER — uncomment and update path when CSV is ready
# schemes_df = spark.read.option("header", "true") \
#                        .option("inferSchema", "true") \
#                        .csv("/Volumes/iitb/govscheme/data/schemes.csv")
#
# schemes_df.write.format("delta") \
#     .mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .saveAsTable("iitb.govscheme.schemes")
#
# print(f"✓ {schemes_df.count()} schemes loaded")
# schemes_df.show(5, truncate=False)



# COMMAND ----------

# Cell 6: Final verification
print("=" * 50)
print("DATA INGESTION — VERIFICATION")
print("=" * 50)

# Profiles
p_count = spark.sql("SELECT COUNT(*) as cnt FROM iitb.govscheme.profiles").collect()[0].cnt
print(f"\n✓ profiles  : {p_count} rows")

# Schemes (will be 0 until Person B loads data)
s_count = spark.sql("SELECT COUNT(*) as cnt FROM iitb.govscheme.schemes").collect()[0].cnt
print(f"✓ schemes   : {s_count} rows {'⏳ waiting for data' if s_count == 0 else ''}")

# Partition check on profiles
print("\nProfiles by state (partition distribution):")
spark.sql("""
    SELECT state, COUNT(*) as count 
    FROM iitb.govscheme.profiles 
    GROUP BY state 
    ORDER BY count DESC
""").show()

print("=" * 50)
print("INGESTION COMPLETE — SCHEMES PENDING PERSON B")
print("=" * 50)

# COMMAND ----------

# Create catalog and schema if they don't exist
spark.sql("CREATE CATALOG IF NOT EXISTS iitb")
spark.sql("CREATE SCHEMA IF NOT EXISTS iitb.govscheme")

# Create a Volume (this is where your files will live)
spark.sql("CREATE VOLUME IF NOT EXISTS iitb.govscheme.raw_data")

print("✅ Volume created at: /Volumes/iitb/govscheme/raw_data/")

# COMMAND ----------

# This replaces the old Cell 1
files = dbutils.fs.ls("/Volumes/iitb/govscheme/raw_data/")
for f in files:
    print(f"  {f.name}  ({f.size} bytes)")

# COMMAND ----------

# Load the LIST file (basic info)
df_list = spark.read.option("multiLine", True).json(
    "/Volumes/iitb/govscheme/raw_data/all_schemes_list.json"
)
print("=== LIST SCHEMA ===")
df_list.printSchema()
print(f"Row count: {df_list.count()}")
df_list.show(5, truncate=False)

# COMMAND ----------

# Load the FULL DATA file (detailed info)
df_full = spark.read.option("multiLine", True).json(
    "/Volumes/iitb/govscheme/raw_data/all_schemes_full_data.json"
)
print("=== FULL DATA SCHEMA ===")
df_full.printSchema()
print(f"Row count: {df_full.count()}")
df_full.show(3, truncate=False)

# COMMAND ----------

# Peek at raw JSON structure of full data
raw_peek = dbutils.fs.head("/Volumes/iitb/govscheme/raw_data/all_schemes_full_data.json", 2000)
print(raw_peek)

# COMMAND ----------

# Same for list
raw_peek_list = dbutils.fs.head("/Volumes/iitb/govscheme/raw_data/all_schemes_list.json", 2000)
print(raw_peek_list)

# COMMAND ----------

# CELL A: Read the full data JSON with Python file I/O (CORRECT way)
import json

# Direct file read — works on Unity Catalog Volumes
with open("/Volumes/iitb/govscheme/raw_data/all_schemes_full_data.json", "r") as f:
    full_data = json.load(f)

print(f"✅ Total records in full data: {len(full_data)}")
print(f"First record keys: {full_data[0].keys()}")
print(f"Status of first record: {full_data[0].get('status')}")

# COMMAND ----------

# CELL B: Recursive function to extract ALL text from nested rich-text blocks

def extract_text_recursive(obj):
    """
    Recursively extracts all 'text' values from deeply nested 
    MyScheme rich-text JSON blocks.
    Works on dicts, lists, and nested children arrays.
    """
    texts = []
    
    if isinstance(obj, str):
        return obj.strip()
    
    if isinstance(obj, list):
        for item in obj:
            result = extract_text_recursive(item)
            if result:
                texts.append(result)
        return " ".join(texts)
    
    if isinstance(obj, dict):
        # If this dict has a "text" key, grab it
        if "text" in obj and obj["text"]:
            texts.append(str(obj["text"]).strip())
        
        # Recurse into "children" if present
        if "children" in obj and obj["children"]:
            result = extract_text_recursive(obj["children"])
            if result:
                texts.append(result)
        
        # Also recurse into "process" (for applicationProcess)
        if "process" in obj and obj["process"]:
            result = extract_text_recursive(obj["process"])
            if result:
                texts.append(result)
                
        # Recurse into list values
        for key, value in obj.items():
            if key not in ["text", "children", "process"] and isinstance(value, (list, dict)):
                result = extract_text_recursive(value)
                if result:
                    texts.append(result)
    
    return " ".join(texts)

# Quick test
test_nested = {"children": [{"text": "Hello"}, {"children": [{"text": "World"}]}]}
print(f"Test: {extract_text_recursive(test_nested)}")
# Should print: "Hello World"

# COMMAND ----------

# CELL C: Extract structured fields from each full_data record

extracted_records = []
errors = 0

for i, record in enumerate(full_data):
    try:
        # Skip failed API responses
        if record.get("status") != "Success":
            errors += 1
            continue
        
        data = record.get("data", {})
        en = data.get("en", {}) if isinstance(data, dict) else {}
        
        if not en:
            errors += 1
            continue
        
        # --- BASIC DETAILS ---
        basic = en.get("basicDetails", {}) or {}
        
        scheme_slug = data.get("slug", "") or record.get("schemeSlug", "") or ""
        scheme_name = basic.get("schemeName", "") or ""
        
        # Ministry
        ministry_obj = basic.get("nodalMinistryName", {}) or {}
        ministry = ministry_obj.get("label", "") if isinstance(ministry_obj, dict) else str(ministry_obj)
        
        # Department
        dept_obj = basic.get("nodalDepartmentName", {}) or {}
        department = dept_obj.get("label", "") if isinstance(dept_obj, dict) else str(dept_obj)
        
        # Tags
        tags_list = basic.get("tags", []) or []
        tags = ", ".join([t if isinstance(t, str) else t.get("label", "") for t in tags_list])
        
        # Target beneficiaries
        target_list = basic.get("targetBeneficiaries", []) or []
        target_beneficiaries = ", ".join([
            t if isinstance(t, str) else t.get("label", "") 
            for t in target_list
        ])
        
        # Scheme sub-categories
        subcat_list = basic.get("schemeSubCategory", []) or []
        sub_categories = ", ".join([
            s if isinstance(s, str) else s.get("label", "") 
            for s in subcat_list
        ])
        
        # --- DESCRIPTION / ABOUT ---
        description_raw = en.get("schemeContent", {}) or {}
        if not description_raw:
            description_raw = en.get("description", {}) or {}
        description = extract_text_recursive(description_raw)[:2000]  # Cap length
        
        # --- BENEFITS ---
        benefits_raw = en.get("benefits", []) or []
        benefits_text = extract_text_recursive(benefits_raw)[:1500]
        
        # --- ELIGIBILITY ---
        eligibility_raw = en.get("eligibility", []) or []
        eligibility_text = extract_text_recursive(eligibility_raw)[:1500]
        
        # --- APPLICATION PROCESS ---
        app_process_raw = en.get("applicationProcess", []) or []
        application_process = extract_text_recursive(app_process_raw)[:1000]
        
        # --- DOCUMENTS REQUIRED ---
        documents_raw = en.get("documents", []) or []
        documents = extract_text_recursive(documents_raw)[:1000]
        
        extracted_records.append({
            "schemeSlug": scheme_slug,
            "schemeName_full": scheme_name,
            "ministry_full": ministry,
            "department": department,
            "tags": tags,
            "target_beneficiaries": target_beneficiaries,
            "sub_categories": sub_categories,
            "description": description if description else "Not available",
            "benefits_text": benefits_text if benefits_text else "Not specified",
            "eligibility_text": eligibility_text if eligibility_text else "Not specified",
            "application_process": application_process if application_process else "Not specified",
            "documents_required": documents if documents else "Not specified",
        })
        
    except Exception as e:
        errors += 1
        if errors <= 5:  # Print first 5 errors only
            print(f"Error on record {i}: {e}")

print(f"✅ Successfully extracted: {len(extracted_records)} schemes")
print(f"⚠️ Errors/skipped: {errors}")

# Quick peek
if extracted_records:
    sample = extracted_records[0]
    for k, v in sample.items():
        print(f"  {k}: {str(v)[:100]}")

# COMMAND ----------

# CELL D: Convert extracted records to Spark DataFrame
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("schemeSlug", StringType(), True),
    StructField("schemeName_full", StringType(), True),
    StructField("ministry_full", StringType(), True),
    StructField("department", StringType(), True),
    StructField("tags", StringType(), True),
    StructField("target_beneficiaries", StringType(), True),
    StructField("sub_categories", StringType(), True),
    StructField("description", StringType(), True),
    StructField("benefits_text", StringType(), True),
    StructField("eligibility_text", StringType(), True),
    StructField("application_process", StringType(), True),
    StructField("documents_required", StringType(), True),
])

df_full_clean = spark.createDataFrame(extracted_records, schema=schema)

print(f"✅ Full data DataFrame: {df_full_clean.count()} rows")
df_full_clean.show(5, truncate=80)

# COMMAND ----------

# CELL E0: Reload list data (in case kernel was restarted)
from pyspark.sql.functions import col, trim, lower

df_list = spark.read.option("multiLine", True).json(
    "/Volumes/iitb/govscheme/raw_data/all_schemes_list.json"
)

print(f"✅ List data reloaded: {df_list.count()} rows")
df_list.show(3, truncate=60)

# COMMAND ----------

# CELL E: Join list + full data using schemeSlug (SERVERLESS COMPATIBLE)
from pyspark.sql.functions import col, trim, lower, count

# Reload list data
df_list = spark.read.option("multiLine", True).json(
    "/Volumes/iitb/govscheme/raw_data/all_schemes_list.json"
)

df_list_clean = df_list.withColumn("slug_key", trim(lower(col("schemeSlug"))))
df_full_clean2 = df_full_clean.withColumn("slug_key", trim(lower(col("schemeSlug"))))

# Diagnostic — using DataFrame ops instead of RDD
list_count = df_list_clean.select("slug_key").distinct().count()
full_count = df_full_clean2.select("slug_key").distinct().count()

# Find overlap using inner join
overlap_df = df_list_clean.select("slug_key").distinct().join(
    df_full_clean2.select("slug_key").distinct(),
    on="slug_key",
    how="inner"
)
overlap_count = overlap_df.count()

print(f"List slugs: {list_count}")
print(f"Full data slugs: {full_count}")
print(f"✅ Overlapping slugs: {overlap_count}")

if overlap_count == 0:
    print("⚠️ NO OVERLAP — checking samples...")
    print("List samples:")
    df_list_clean.select("slug_key").show(5, truncate=False)
    print("Full data samples:")
    df_full_clean2.select("slug_key").show(5, truncate=False)

# COMMAND ----------

# CELL E_COMBINED FIXED: Load, Join, Enrich — ALL IN ONE
from pyspark.sql.functions import (
    col, trim, lower, rand, floor, array, element_at, 
    lit, when, concat_ws
)
import pyspark.sql.functions as F

# --- STEP 1: Reload list data ---
df_list = spark.read.option("multiLine", True).json(
    "/Volumes/iitb/govscheme/raw_data/all_schemes_list.json"
)
print(f"List rows: {df_list.count()}")

# --- STEP 2: Prepare join keys ---
df_list_clean = df_list.withColumn("slug_key", trim(lower(col("schemeSlug"))))
df_full_clean2 = df_full_clean.withColumn("slug_key", trim(lower(col("schemeSlug"))))

# --- STEP 3: Check overlap ---
overlap_count = df_list_clean.select("slug_key").distinct().join(
    df_full_clean2.select("slug_key").distinct(),
    on="slug_key",
    how="inner"
).count()

print(f"✅ Overlapping slugs: {overlap_count}")

# --- STEP 4: Join (FIXED) ---
# Rename columns in full_clean to avoid conflicts, then join on slug_key
# Keep slug_key in BOTH dataframes, use list syntax for join condition

df_merged = df_list_clean.alias("L").join(
    df_full_clean2.alias("R"),
    col("L.slug_key") == col("R.slug_key"),
    how="inner"
).select(
    # From LIST (basic info)
    col("L.schemeId"),
    col("L.schemeName"),
    col("L.schemeSlug"),
    col("L.category"),
    col("L.ministry"),
    col("L.index"),
    # From FULL DATA (detailed info)
    col("R.description"),
    col("R.benefits_text"),
    col("R.eligibility_text"),
    col("R.application_process"),
    col("R.documents_required"),
    col("R.tags"),
    col("R.target_beneficiaries"),
    col("R.sub_categories"),
    col("R.department"),
    col("R.schemeName_full"),
    col("R.ministry_full"),
)

print(f"✅ Merged (inner join): {df_merged.count()} rows")
df_merged.show(3, truncate=60)

# --- STEP 5: Add synthetic eligibility columns ---
SEED = 42

df_enriched = df_merged \
    .withColumn("income_max_lpa",
        element_at(
            array(*[lit(float(x)) for x in [1.0, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 25.0]]),
            (floor(rand(SEED) * 10) + 1).cast("int")
        )
    ) \
    .withColumn("age_min",
        element_at(
            array(*[lit(x) for x in [0, 0, 14, 18, 18, 21, 21, 60]]),
            (floor(rand(SEED + 1) * 8) + 1).cast("int")
        )
    ) \
    .withColumn("age_max",
        element_at(
            array(*[lit(x) for x in [18, 25, 35, 45, 59, 60, 80, 100]]),
            (floor(rand(SEED + 2) * 8) + 1).cast("int")
        )
    ) \
    .withColumn("occupation",
        element_at(
            array(*[lit(x) for x in ["farmer", "student", "unemployed", "employed", "self-employed", "any", "any", "any"]]),
            (floor(rand(SEED + 3) * 8) + 1).cast("int")
        )
    ) \
    .withColumn("caste_eligible",
        element_at(
            array(*[lit(x) for x in ["all", "sc", "st", "obc", "sc,st", "sc,st,obc", "all", "all"]]),
            (floor(rand(SEED + 4) * 8) + 1).cast("int")
        )
    ) \
    .withColumn("benefit_inr",
        element_at(
            array(*[lit(x) for x in [5000, 6000, 12000, 25000, 50000, 75000, 100000, 200000, 500000, 1000000]]),
            (floor(rand(SEED + 5) * 10) + 1).cast("int")
        )
    ) \
    .withColumn("gender_eligible",
        element_at(
            array(*[lit(x) for x in ["all", "all", "all", "female", "all", "male", "all", "all"]]),
            (floor(rand(SEED + 6) * 8) + 1).cast("int")
        )
    ) \
    .withColumn("is_rural",
        when(rand(SEED + 7) > 0.7, 1).otherwise(0)
    ) \
    .withColumn("state_eligible",
        element_at(
            array(*[lit(x) for x in ["all", "all", "all", "maharashtra", "karnataka", "up", "mp", "all"]]),
            (floor(rand(SEED + 8) * 8) + 1).cast("int")
        )
    )

# Fix age_min < age_max
df_enriched = df_enriched.withColumn(
    "age_max",
    when(col("age_max") <= col("age_min"), col("age_min") + 20).otherwise(col("age_max"))
)

# Convert category array to string
df_enriched = df_enriched.withColumn(
    "category_str",
    concat_ws(", ", col("category"))
)

print(f"\n✅ FINAL enriched dataset: {df_enriched.count()} rows")
print("\n=== SAMPLE DATA ===")
df_enriched.select(
    "schemeName", "income_max_lpa", "age_min", "age_max",
    "occupation", "caste_eligible", "benefit_inr", "gender_eligible"
).show(10, truncate=40)

# COMMAND ----------

# CELL G: Save to Delta
spark.sql("CREATE CATALOG IF NOT EXISTS iitb")
spark.sql("CREATE SCHEMA IF NOT EXISTS iitb.govscheme")

df_enriched.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("iitb.govscheme.schemes")

df_verify = spark.table("iitb.govscheme.schemes")
print(f"✅ Delta table saved: {df_verify.count()} rows")

# COMMAND ----------

# CELL H: Test eligibility query
eligible_df = spark.sql("""
    SELECT schemeName, benefit_inr, occupation, caste_eligible, age_min, age_max
    FROM iitb.govscheme.schemes
    WHERE income_max_lpa >= 3.0
      AND age_min <= 25
      AND age_max >= 25
      AND (occupation = 'farmer' OR occupation = 'any')
      AND (caste_eligible LIKE '%obc%' OR caste_eligible = 'all')
      AND (gender_eligible = 'male' OR gender_eligible = 'all')
    ORDER BY benefit_inr DESC
""")

print(f"✅ Eligible schemes: {eligible_df.count()}")
eligible_df.show(20, truncate=60)

# COMMAND ----------

import os

APP_DIR = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"
app_py_path = os.path.join(APP_DIR, "app.py")

with open(app_py_path, 'r') as f:
    content = f.read()

# Find the /api/process_text endpoint definition
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'process_text' in line or 'ProcessText' in line or 'class.*Request' in line.lower():
        # Show surrounding context
        start = max(0, i-2)
        end = min(len(lines), i+20)
        for j in range(start, end):
            print(f"{j+1:4d} | {lines[j]}")
        print("---")

# COMMAND ----------

{
  "text": "...",
  "input_language": "hindi",
  "output_language": "auto",
  "max_schemes": 10,
  "generate_audio": true
}

# COMMAND ----------

import os

APP_DIR = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"
app_py_path = os.path.join(APP_DIR, "app.py")

with open(app_py_path, 'r') as f:
    content = f.read()

# Replace the process_text endpoint to accept JSON body
old_endpoint = '''@app.post("/api/process_text")
async def process_text(
    text: str = Form(...),
    input_language: str = Form("hindi"),
    output_language: str = Form("auto"),
    max_schemes: int = Form(10)
):
    return run_pipeline(text, input_language, output_language, max_schemes, is_audio=False)'''

new_endpoint = '''@app.post("/api/process_text")
async def process_text(request: Request):
    body = await request.json()
    text = body.get("text", "")
    input_language = body.get("input_language", "hindi")
    output_language = body.get("output_language", "auto")
    max_schemes = int(body.get("max_schemes", 10))
    generate_audio = body.get("generate_audio", True)
    return run_pipeline(text, input_language, output_language, max_schemes, is_audio=False)'''

if old_endpoint in content:
    content = content.replace(old_endpoint, new_endpoint)
    print("✅ Replaced process_text endpoint (Form → JSON)")
else:
    print("⚠️ Exact match not found, trying flexible match...")
    # Try line by line
    lines = content.split('\n')
    new_lines = []
    skip_until_return = False
    replaced = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if '@app.post("/api/process_text")' in line and not replaced:
            # Replace this entire block
            new_lines.append('@app.post("/api/process_text")')
            new_lines.append('async def process_text(request: Request):')
            new_lines.append('    body = await request.json()')
            new_lines.append('    text = body.get("text", "")')
            new_lines.append('    input_language = body.get("input_language", "hindi")')
            new_lines.append('    output_language = body.get("output_language", "auto")')
            new_lines.append('    max_schemes = int(body.get("max_schemes", 10))')
            new_lines.append('    generate_audio = body.get("generate_audio", True)')
            new_lines.append('    return run_pipeline(text, input_language, output_language, max_schemes, is_audio=False)')
            # Skip old lines until we find the return or next decorator
            i += 1
            while i < len(lines):
                if 'return run_pipeline' in lines[i]:
                    i += 1
                    break
                i += 1
            replaced = True
            continue
        new_lines.append(line)
        i += 1
    
    if replaced:
        content = '\n'.join(new_lines)
        print("✅ Replaced process_text endpoint (flexible match)")
    else:
        print("❌ Could not find endpoint to replace!")

# Also make sure Request is imported from fastapi
if 'from fastapi import' in content and 'Request' not in content.split('from fastapi import')[1].split('\n')[0]:
    content = content.replace('from fastapi import', 'from fastapi import Request, ', 1)
    print("✅ Added Request import")
elif 'Request' in content:
    print("✅ Request already imported")

# Write back
with open(app_py_path, 'w') as f:
    f.write(content)

print(f"\n📄 app.py updated: {len(content)} chars")
print("\n🚀 Now redeploy the app from Databricks Apps UI!")

# COMMAND ----------


# Databricks notebook source
# Cell 1: Setup
spark.sql("USE CATALOG iitb")
spark.sql("USE SCHEMA govscheme")
print(f"✓ Spark {spark.version} ready")
print("✓ Using iitb.govscheme")

# COMMAND ----------

# Cell 2: Core eligibility filter function (upgraded)
def get_eligible_schemes(profile: dict):
    query = f"""
        SELECT
            scheme_id,
            name,
            category,
            state,
            benefit_inr,
            benefit_type,
            description,
            application_url
        FROM iitb.govscheme.schemes
        WHERE
            (income_max_lpa IS NULL OR income_max_lpa >= {profile['income_lpa']})
            AND (age_min IS NULL OR age_min <= {profile['age']})
            AND (age_max IS NULL OR age_max >= {profile['age']})
            AND (gender IS NULL OR gender = 'Any' OR gender = '{profile['gender']}')
            AND (state = 'All' OR state = '{profile['state']}')
            AND (occupation IS NULL OR occupation = 'Any'
                 OR occupation = '{profile['occupation']}')
            AND (caste_eligible IS NULL OR caste_eligible = 'Any'
                 OR caste_eligible = '{profile['caste']}')
        ORDER BY benefit_inr DESC NULLS LAST
    """
    return spark.sql(query)

print("✓ get_eligible_schemes() defined")

# COMMAND ----------

# Cell 3: Demo persona — Ramesh (this is who we demo to judges)
DEMO_PROFILE = {
    "profile_id"      : "demo-ramesh-001",
    "age"             : 45,
    "gender"          : "Male",
    "state"           : "Rajasthan",
    "income_lpa"      : 0.8,
    "occupation"      : "Farmer",
    "caste"           : "OBC",
    "has_bpl_card"    : "Yes",
    "land_owned_acres": 2.5
}

print("Demo persona locked:")
for k, v in DEMO_PROFILE.items():
    print(f"   {k:20s}: {v}")

# COMMAND ----------

# Cell 4: Test engine against Ramesh
result_df = get_eligible_schemes(DEMO_PROFILE)
count = result_df.count()

print(f"Eligible schemes for Ramesh: {count}")
if count == 0:
    print("⏳ No schemes yet — engine is correct, data pending from Person B")
else:
    result_df.show(truncate=False)

# COMMAND ----------


# Cell 5: Batch eligibility — distributed Spark join across all 500 profiles
from pyspark.sql.functions import col

def run_batch_eligibility():
    """
    Cross join profiles x schemes, apply all rules as Spark column
    expressions. Fully distributed — no Python UDF. Pure Spark.
    """
    profiles_df = spark.table("iitb.govscheme.profiles")
    schemes_df  = spark.table("iitb.govscheme.schemes")

    matches = (profiles_df.alias("p")
        .crossJoin(schemes_df.alias("s"))
        .filter(
            (col("s.income_max_lpa").isNull() |
             (col("s.income_max_lpa") >= col("p.income_lpa")))
            & (col("s.age_min").isNull() |
               (col("s.age_min") <= col("p.age")))
            & (col("s.age_max").isNull() |
               (col("s.age_max") >= col("p.age")))
            & (col("s.gender").isNull() |
               (col("s.gender") == "Any") |
               (col("s.gender") == col("p.gender")))
            & ((col("s.state") == "All") |
               (col("s.state") == col("p.state")))
            & (col("s.occupation").isNull() |
               (col("s.occupation") == "Any") |
               (col("s.occupation") == col("p.occupation")))
            & (col("s.caste_eligible").isNull() |
               (col("s.caste_eligible") == "Any") |
               (col("s.caste_eligible") == col("p.caste")))
        )
        .select(
            col("p.profile_id"),
            col("p.occupation"),
            col("p.state").alias("user_state"),
            col("s.scheme_id"),
            col("s.name").alias("scheme_name"),
            col("s.category"),
            col("s.benefit_inr"),
            col("s.benefit_type"),
        )
    )
    return matches

print("✓ run_batch_eligibility() defined")
print("⏳ Run after Person B loads scheme data")

# COMMAND ----------

# Cell 6: Save batch results to Delta (uncomment after data loads)

# matches_df = run_batch_eligibility()
#
# matches_df.write.format("delta") \
#     .mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .saveAsTable("iitb.govscheme.eligibility_results")
#
# total    = matches_df.count()
# profiles = matches_df.select("profile_id").distinct().count()
# print(f"✓ {total} matches written to iitb.govscheme.eligibility_results")
# print(f"✓ {profiles} profiles have at least one eligible scheme")
# matches_df.groupBy("category").count().orderBy("count", ascending=False).show()

print("✓ Cell 6 ready — uncomment after scheme data loads")

# COMMAND ----------

# Cell 7: Verification
print("=" * 50)
print("ELIGIBILITY ENGINE — VERIFICATION")
print("=" * 50)

s_count = spark.sql("SELECT COUNT(*) as cnt FROM iitb.govscheme.schemes").collect()[0].cnt
p_count = spark.sql("SELECT COUNT(*) as cnt FROM iitb.govscheme.profiles").collect()[0].cnt

print(f"\n✓ schemes table            : {s_count} rows {'⏳ pending' if s_count == 0 else ''}")
print(f"✓ profiles table           : {p_count} rows")
print(f"✓ get_eligible_schemes()   : single profile filter  — ready")
print(f"✓ run_batch_eligibility()  : distributed Spark join — ready")
print(f"✓ DEMO_PROFILE (Ramesh)    : locked in")

print("\n" + "=" * 50)
print("ENGINE COMPLETE — READY TO PLUG IN DATA")
print("=" * 50)

# COMMAND ----------

# CELL 1: Load the schemes Delta table
from pyspark.sql.functions import (
    col, lit, when, lower, array_contains, 
    expr, concat_ws, row_number, desc, asc
)
from pyspark.sql.window import Window

# Load our enriched schemes table
df_schemes = spark.table("iitb.govscheme.schemes")
print(f"✅ Loaded {df_schemes.count()} schemes from Delta table")
print(f"Columns: {df_schemes.columns}")

# COMMAND ----------

# CELL 2: Core eligibility engine

def get_eligible_schemes(user_profile: dict) -> "DataFrame":
    """
    Filters government schemes based on user profile.
    
    Parameters:
        user_profile: dict with keys:
            - age (int)
            - income_lpa (float)
            - occupation (str): farmer/student/unemployed/employed/self-employed
            - caste (str): general/sc/st/obc
            - gender (str): male/female/other
            - is_rural (int): 0 or 1
            - state (str): state name or "all"
            - disability (str, optional): "yes" or "no"
    
    Returns:
        Spark DataFrame of eligible schemes with match scores
    """
    
    df = spark.table("iitb.govscheme.schemes")
    
    age = user_profile.get("age", 25)
    income = user_profile.get("income_lpa", 3.0)
    occupation = user_profile.get("occupation", "any").lower().strip()
    caste = user_profile.get("caste", "general").lower().strip()
    gender = user_profile.get("gender", "all").lower().strip()
    is_rural = user_profile.get("is_rural", 0)
    state = user_profile.get("state", "all").lower().strip()
    
    # ===== HARD FILTERS (must match) =====
    df_eligible = df.filter(
        # Age filter
        (col("age_min") <= age) & (col("age_max") >= age)
    ).filter(
        # Income filter
        (col("income_max_lpa") >= income)
    ).filter(
        # Occupation filter
        (lower(col("occupation")) == occupation) | 
        (lower(col("occupation")) == "any")
    ).filter(
        # Gender filter
        (lower(col("gender_eligible")) == gender) | 
        (lower(col("gender_eligible")) == "all")
    ).filter(
        # Caste filter
        (lower(col("caste_eligible")).contains(caste)) | 
        (lower(col("caste_eligible")) == "all")
    )
    
    # ===== SOFT SCORING (adds relevance points) =====
    df_scored = df_eligible.withColumn(
        "match_score",
        lit(0)
        # Exact occupation match = +20 points
        + when(lower(col("occupation")) == occupation, 20).otherwise(0)
        # Exact caste match = +15 points  
        + when(lower(col("caste_eligible")).contains(caste), 15).otherwise(0)
        # Rural match = +10 points
        + when(
            (col("is_rural") == 1) & (lit(is_rural) == 1), 10
        ).otherwise(0)
        # State match = +15 points
        + when(
            (lower(col("state_eligible")) == state) | 
            (lower(col("state_eligible")) == "all"), 15
        ).otherwise(0)
        # Higher benefit = more points (normalized)
        + when(col("benefit_inr") >= 100000, 20)
         .when(col("benefit_inr") >= 50000, 15)
         .when(col("benefit_inr") >= 25000, 10)
         .when(col("benefit_inr") >= 10000, 5)
         .otherwise(0)
        # Category relevance bonus
        + when(col("category_str").contains("Education") & (lit(occupation) == "student"), 10).otherwise(0)
        + when(col("category_str").contains("Agriculture") & (lit(occupation) == "farmer"), 10).otherwise(0)
        + when(col("category_str").contains("Business") & (lit(occupation) == "self-employed"), 10).otherwise(0)
    )
    
    # ===== SELECT OUTPUT COLUMNS =====
    df_result = df_scored.select(
        "schemeId",
        "schemeName",
        "schemeSlug",
        "category_str",
        "ministry",
        "description",
        "benefits_text",
        "eligibility_text",
        "application_process",
        "benefit_inr",
        "income_max_lpa",
        "age_min",
        "age_max",
        "occupation",
        "caste_eligible",
        "gender_eligible",
        "state_eligible",
        "is_rural",
        "tags",
        "target_beneficiaries",
        "match_score"
    ).orderBy(desc("match_score"), desc("benefit_inr"))
    
    return df_result

print("✅ Eligibility engine function defined")

# COMMAND ----------

# CELL 3: Test with multiple user profiles

# ===== TEST USER 1: Young Farmer =====
user_farmer = {
    "age": 25,
    "income_lpa": 2.5,
    "occupation": "farmer",
    "caste": "obc",
    "gender": "male",
    "is_rural": 1,
    "state": "maharashtra"
}

df_farmer = get_eligible_schemes(user_farmer)
farmer_count = df_farmer.count()
print(f"🌾 Young Farmer — Eligible schemes: {farmer_count}")
df_farmer.select("schemeName", "benefit_inr", "match_score", "category_str").show(10, truncate=50)

# ===== TEST USER 2: Female Student =====
user_student = {
    "age": 20,
    "income_lpa": 2.0,
    "occupation": "student",
    "caste": "sc",
    "gender": "female",
    "is_rural": 0,
    "state": "karnataka"
}

df_student = get_eligible_schemes(user_student)
student_count = df_student.count()
print(f"👩‍🎓 Female SC Student — Eligible schemes: {student_count}")
df_student.select("schemeName", "benefit_inr", "match_score", "category_str").show(10, truncate=50)

# ===== TEST USER 3: Senior Citizen =====
user_senior = {
    "age": 65,
    "income_lpa": 1.5,
    "occupation": "unemployed",
    "caste": "general",
    "gender": "male",
    "is_rural": 1,
    "state": "up"
}

df_senior = get_eligible_schemes(user_senior)
senior_count = df_senior.count()
print(f"👴 Senior Citizen — Eligible schemes: {senior_count}")
df_senior.select("schemeName", "benefit_inr", "match_score", "category_str").show(10, truncate=50)

print(f"""
╔══════════════════════════════════════╗
║   ELIGIBILITY ENGINE SUMMARY        ║
╠══════════════════════════════════════╣
║  Young Farmer:    {farmer_count:>4} schemes       ║
║  Female Student:  {student_count:>4} schemes       ║
║  Senior Citizen:  {senior_count:>4} schemes       ║
╚══════════════════════════════════════╝
""")

# COMMAND ----------

# CELL 4: Parse natural language profile into structured dict

def parse_user_profile(text: str) -> dict:
    """
    Parses a natural language user description into structured profile.
    Simple keyword extraction — fast and reliable for demo.
    
    Example input: "I am a 25 year old female farmer from Maharashtra, 
                    income 2 lakh, OBC category, living in rural area"
    """
    import re
    text_lower = text.lower()
    
    # --- AGE ---
    age_match = re.search(r'(\d{1,3})\s*(?:year|yr|age)', text_lower)
    if not age_match:
        age_match = re.search(r'age\s*[:\-]?\s*(\d{1,3})', text_lower)
    age = int(age_match.group(1)) if age_match else 30
    
    # --- INCOME ---
    income = 3.0  # default
    income_match = re.search(r'(\d+\.?\d*)\s*(?:lakh|lac|lpa|l)', text_lower)
    if income_match:
        income = float(income_match.group(1))
    income_match2 = re.search(r'income\s*[:\-]?\s*(\d+\.?\d*)', text_lower)
    if income_match2:
        income = float(income_match2.group(1))
    # If someone says "50000" or "50,000" (in rupees, convert to LPA)
    income_match3 = re.search(r'(\d{4,7})\s*(?:per\s*(?:year|annum|month))?', text_lower)
    if income_match3 and not income_match:
        val = float(income_match3.group(1).replace(',', ''))
        if val > 100:  # likely in rupees
            if 'month' in text_lower:
                income = (val * 12) / 100000
            else:
                income = val / 100000
    
    # --- OCCUPATION ---
    occupation = "any"
    occupation_keywords = {
        "farmer": ["farmer", "farming", "agriculture", "kisan", "krishi"],
        "student": ["student", "studying", "college", "university", "school", "vidyarthi"],
        "unemployed": ["unemployed", "jobless", "no job", "berozgar", "no work", "no employment"],
        "employed": ["employed", "working", "job", "service", "salaried", "naukri"],
        "self-employed": ["self-employed", "self employed", "business", "shop", "entrepreneur", "vyapari", "startup"],
    }
    for occ, keywords in occupation_keywords.items():
        if any(kw in text_lower for kw in keywords):
            occupation = occ
            break
    
    # --- CASTE ---
    caste = "general"
    if any(x in text_lower for x in ["obc", "other backward"]):
        caste = "obc"
    elif any(x in text_lower for x in [" st ", "scheduled tribe", "tribal", "adivasi"]):
        caste = "st"
    elif any(x in text_lower for x in [" sc ", "scheduled caste", "dalit"]):
        caste = "sc"
    elif any(x in text_lower for x in ["general", "unreserved"]):
        caste = "general"
    
    # --- GENDER ---
    gender = "male"
    if any(x in text_lower for x in ["female", "woman", "girl", "mahila", "lady", "she ", "her "]):
        gender = "female"
    elif any(x in text_lower for x in ["male", "man", "boy", "he ", "his "]):
        gender = "male"
    
    # --- RURAL/URBAN ---
    is_rural = 0
    if any(x in text_lower for x in ["rural", "village", "gram", "gaon"]):
        is_rural = 1
    elif any(x in text_lower for x in ["urban", "city", "town", "shahar", "nagar"]):
        is_rural = 0
    
    # --- STATE ---
    state = "all"
    states_map = {
        "maharashtra": ["maharashtra", "mumbai", "pune", "nagpur"],
        "karnataka": ["karnataka", "bangalore", "bengaluru", "mysore"],
        "up": ["uttar pradesh", "up ", "lucknow", "noida"],
        "mp": ["madhya pradesh", "mp ", "bhopal", "indore"],
        "rajasthan": ["rajasthan", "jaipur", "jodhpur"],
        "tamil_nadu": ["tamil nadu", "chennai", "coimbatore"],
        "kerala": ["kerala", "kochi", "trivandrum"],
        "bihar": ["bihar", "patna"],
        "west_bengal": ["west bengal", "kolkata"],
        "gujarat": ["gujarat", "ahmedabad", "surat"],
        "delhi": ["delhi", "new delhi"],
        "telangana": ["telangana", "hyderabad"],
        "andhra_pradesh": ["andhra pradesh", "vijayawada", "visakhapatnam"],
        "punjab": ["punjab", "chandigarh", "ludhiana"],
        "haryana": ["haryana", "gurgaon", "gurugram"],
        "jharkhand": ["jharkhand", "ranchi"],
        "odisha": ["odisha", "orissa", "bhubaneswar"],
        "assam": ["assam", "guwahati"],
        "goa": ["goa", "panaji"],
    }
    for state_key, keywords in states_map.items():
        if any(kw in text_lower for kw in keywords):
            state = state_key
            break
    
    profile = {
        "age": age,
        "income_lpa": income,
        "occupation": occupation,
        "caste": caste,
        "gender": gender,
        "is_rural": is_rural,
        "state": state,
    }
    
    return profile

print("✅ Profile parser defined")

# COMMAND ----------

# CELL 5: Test parser with realistic inputs

test_inputs = [
    "I am a 25 year old female farmer from Maharashtra, income 2 lakh, OBC category, living in rural area",
    "22 year old male SC student from Karnataka, family income 1.5 lakh per annum, urban",
    "I'm a 45 year old self-employed man from Gujarat, general category, income 8 lakh",
    "65 year old unemployed woman from rural UP, ST community, income below 1 lakh",
    "I am 30 years old farmer from village in Bihar, OBC, male, income 3 lpa",
]

for text in test_inputs:
    profile = parse_user_profile(text)
    eligible = get_eligible_schemes(profile)
    count = eligible.count()
    print(f"\n📝 Input: \"{text[:70]}...\"")
    print(f"   Parsed: age={profile['age']}, income={profile['income_lpa']}, occ={profile['occupation']}, "
          f"caste={profile['caste']}, gender={profile['gender']}, rural={profile['is_rural']}, state={profile['state']}")
    print(f"   ✅ Eligible schemes: {count}")
    if count > 0:
        eligible.select("schemeName", "benefit_inr", "match_score").show(3, truncate=50)

# COMMAND ----------

# CELL 6: Save a sample eligibility run to Delta for ML training

# Use a diverse set of user profiles to generate training data
profiles = [
    {"age": 22, "income_lpa": 1.5, "occupation": "student", "caste": "sc", "gender": "female", "is_rural": 0, "state": "karnataka"},
    {"age": 30, "income_lpa": 3.0, "occupation": "farmer", "caste": "obc", "gender": "male", "is_rural": 1, "state": "maharashtra"},
    {"age": 45, "income_lpa": 5.0, "occupation": "employed", "caste": "general", "gender": "male", "is_rural": 0, "state": "delhi"},
    {"age": 65, "income_lpa": 1.0, "occupation": "unemployed", "caste": "st", "gender": "female", "is_rural": 1, "state": "up"},
    {"age": 28, "income_lpa": 2.0, "occupation": "self-employed", "caste": "obc", "gender": "male", "is_rural": 0, "state": "gujarat"},
    {"age": 19, "income_lpa": 1.0, "occupation": "student", "caste": "sc", "gender": "male", "is_rural": 1, "state": "bihar"},
    {"age": 35, "income_lpa": 4.0, "occupation": "farmer", "caste": "general", "gender": "female", "is_rural": 1, "state": "rajasthan"},
    {"age": 50, "income_lpa": 8.0, "occupation": "self-employed", "caste": "general", "gender": "male", "is_rural": 0, "state": "tamil_nadu"},
]

from pyspark.sql.functions import lit
from functools import reduce

all_eligible = []
for i, profile in enumerate(profiles):
    df_elig = get_eligible_schemes(profile)
    # Add user context for ML training
    df_elig = df_elig \
        .withColumn("user_id", lit(i)) \
        .withColumn("user_age", lit(profile["age"])) \
        .withColumn("user_income", lit(profile["income_lpa"])) \
        .withColumn("user_occupation", lit(profile["occupation"])) \
        .withColumn("user_caste", lit(profile["caste"])) \
        .withColumn("user_gender", lit(profile["gender"])) \
        .withColumn("user_is_rural", lit(profile["is_rural"])) \
        .withColumn("user_state", lit(profile["state"]))
    all_eligible.append(df_elig)

# Union all results
df_training_base = reduce(lambda a, b: a.unionByName(b), all_eligible)

print(f"✅ Training base data: {df_training_base.count()} rows from {len(profiles)} user profiles")

# Save to Delta for ML notebook
df_training_base.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("iitb.govscheme.eligibility_results")

print(f"✅ Saved to iitb.govscheme.eligibility_results")
df_training_base.select("user_id", "schemeName", "match_score", "benefit_inr").show(10, truncate=50)

# COMMAND ----------

# CELL 7: Final verification of everything

print("=" * 60)
print("NOTEBOOK 02: ELIGIBILITY ENGINE — STATUS")
print("=" * 60)

# Check Delta tables
schemes_count = spark.table("iitb.govscheme.schemes").count()
eligibility_count = spark.table("iitb.govscheme.eligibility_results").count()

print(f"""
✅ Schemes Delta table:          {schemes_count} rows
✅ Eligibility results table:    {eligibility_count} rows

Functions available:
  ✅ get_eligible_schemes(user_profile)  → Filtered + scored DataFrame
  ✅ parse_user_profile(text)            → Natural language → profile dict

Pipeline flow:
  User text → parse_user_profile() → get_eligible_schemes() → Ranked results
  
Ready for:
  → Notebook 03 (ML Training)
  → Notebook 05 (Pipeline Integration)
""")

# Quick end-to-end test
test_text = "I am a 28 year old male farmer from rural Maharashtra, OBC, income 2.5 lakh"
profile = parse_user_profile(test_text)
results = get_eligible_schemes(profile)

print(f"🔥 End-to-end test: \"{test_text}\"")
print(f"   → {results.count()} eligible schemes found")
print(f"   → Top 5:")
results.select("schemeName", "benefit_inr", "match_score").show(5, truncate=55)

# COMMAND ----------


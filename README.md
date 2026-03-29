# Yojana AI

AI-powered government scheme discovery platform that helps Indian citizens find eligible schemes through natural language input (text or voice) in 11 Indian regional languages, powered by Databricks, Sarvam AI, and PySpark MLlib.

Given a user profile typed or spoken in any Indian language, Yojana AI identifies eligible schemes from 4787 government programs, ranks them using a trained ML model, optimizes a diverse bundle, and returns results with translated text and voice output.

**Live App:** [https://yojana-ai-7474657458509072.aws.databricksapps.com](https://yojana-ai-7474657458509072.aws.databricksapps.com)

---

## Architecture

```mermaid
flowchart TD
    A["User Input\nText or Voice\n11 Indian Languages"] --> B{"Input Type?"}
    
    B -->|Voice| C["Sarvam STT\nsaarika v2.5\nVoice to Text"]
    B -->|Text| D["Sarvam Translation\nmayura v1\nRegional to English"]
    C --> D

    D --> E["NLP Profile Parser\nMulti-language Regex\nAge, Gender, Income,\nOccupation, Caste, State"]

    subgraph DATABRICKS ["Databricks Platform"]
        E --> F["Apache Spark\nEligibility Engine\nSQL Filters on\nDelta Lake"]
        F --> G["Delta Lake\n4787 Schemes\n6 Tables"]
        G --> H["PySpark MLlib\nGBTRegressor\n22 Features\nR2 = 0.98"]
        H --> I["MLflow\nExperiment Tracking\nModel Logging"]
        H --> J["Bundle Optimizer\nGreedy Diversity\nMax 3 per Category"]
    end

    J --> K["Sarvam Translation\nmayura v1\nEnglish to Regional"]
    J --> L["Sarvam TTS\nbulbul v2\nText to Voice"]

    K --> M["Results\nRanked Schemes\nTranslated Text\nVoice Audio Output"]
    L --> M

    style DATABRICKS fill:#1a1a2e,stroke:#6c5ce7,stroke-width:2px,color:#fff
    style A fill:#6c5ce7,stroke:#6c5ce7,color:#fff
    style M fill:#00b894,stroke:#00b894,color:#fff
    style C fill:#e17055,stroke:#e17055,color:#fff
    style D fill:#0984e3,stroke:#0984e3,color:#fff
    style K fill:#0984e3,stroke:#0984e3,color:#fff
    style L fill:#e17055,stroke:#e17055,color:#fff
    style H fill:#fdcb6e,stroke:#fdcb6e,color:#1a1a2e
    style I fill:#a29bfe,stroke:#a29bfe,color:#1a1a2e
    style F fill:#636e72,stroke:#636e72,color:#fff
    style G fill:#636e72,stroke:#636e72,color:#fff
    style J fill:#636e72,stroke:#636e72,color:#fff
    style E fill:#00cec9,stroke:#00cec9,color:#1a1a2e

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Platform | Databricks Serverless, Unity Catalog |
| Processing | Apache Spark, PySpark |
| Storage | Delta Lake (6 tables) |
| ML | PySpark MLlib GBTRegressor, 22 features |
| Tracking | MLflow |
| AI Models | Sarvam AI saarika v2.5 (STT), mayura v1 (Translation), bulbul v2 (TTS) |
| Backend | FastAPI, Python, Pandas |
| Frontend | HTML5, CSS3, JavaScript |
| Deployment | Databricks Apps |
| Data Source | MyScheme Government of India API |


## Repository Structure

    yojana-ai/
    ├── README.md
    ├── Raw_data/
    │   ├── all_schemes_list.json
    │   └── all_schemes_full_data.json
    ├── notebooks/
    │   ├── 01_data_ingestion.py
    │   ├── 02_eligibility_engine.py
    │   ├── 03_ml_training.py
    │   ├── 04_optimization.py
    │   ├── 05_pipeline_integration.py
    │   └── 06_interactive_demo.py
    └── app/
        ├── app.py
        ├── app.yaml
        ├── requirements.txt
        ├── schemes_data.json
        └── static/
            └── index.html


## How to Run

### Prerequisites

Databricks workspace with Unity Catalog and Serverless compute enabled.
Sarvam AI API key from https://sarvam.ai

### Step 1: Setup Environment

Run in Databricks SQL or notebook:

    CREATE CATALOG IF NOT EXISTS iitb;
    CREATE SCHEMA IF NOT EXISTS iitb.govscheme;

Upload raw JSON files to /Volumes/iitb/govscheme/raw_data/

### Step 2: Run Notebooks in Order

    01_data_ingestion.py       Creates iitb.govscheme.schemes (4787 rows)
    02_eligibility_engine.py   Creates iitb.govscheme.eligibility_results (4879 rows)
    03_ml_training.py          Trains GBTRegressor, logs to MLflow
    04_optimization.py         Creates iitb.govscheme.optimized_bundle
    05_pipeline_integration.py Validates 3 test profiles end-to-end

### Step 3: Export Scheme Data

    df = spark.table("iitb.govscheme.schemes").toPandas()
    df.to_json("app/schemes_data.json", orient="records")

### Step 4: Deploy App

    1. Databricks UI then Apps then Create new app named yojana-ai
    2. Set source to app/ directory
    3. Add environment variable SARVAM_API_KEY with your key
    4. Click Deploy


## Demo Steps

### Text Input

    1. Open the app URL in browser
    2. Click any example preset such as Hindi Farmer or Tamil Student or Marathi Worker
    3. The textarea fills with regional language text and input language auto-selects
    4. Click Find Eligible Schemes
    5. Wait 10-15 seconds for processing
    6. Results appear with auto-playing voice summary in detected regional language
    7. Each scheme card shows translated name and description
    8. Click Translate on any section to see regional language text
    9. Click Listen on any section to hear it spoken in regional language

### Voice Input

    1. Switch to Voice Input tab
    2. Select your language from dropdown
    3. Click microphone button and speak your profile
    4. Click stop when done
    5. Preview recording with playback player
    6. Click Process Voice Input
    7. Results appear with voice output in your language

### Test Profiles

| Profile | Language | Expected Output |
|---------|----------|----------------|
| 35M Farmer, UP, OBC, Rural, 2L income | Hindi | Hindi |
| 20F Student, Tamil Nadu, SC, Rural, 1L | Tamil | Tamil |
| 40M Worker, Maharashtra, OBC, Urban, 3L | Marathi | Marathi |
| 30F Woman, Andhra Pradesh, OBC, Rural, 1.5L | Telugu | Telugu |
| 19M Student, West Bengal, SC, Rural, 1L | Bengali | Bengali |
| 68M Retired, Delhi, General, Urban, 3L | English | Hindi |


## Delta Lake Tables

| Table | Rows | Purpose |
|-------|------|---------|
| iitb.govscheme.schemes | 4787 | Master scheme data with eligibility columns |
| iitb.govscheme.eligibility_results | 4879 | ML training data from user-scheme pairs |
| iitb.govscheme.feature_importance | 22 | GBT model feature weights |
| iitb.govscheme.ranked_results | 891 | ML-ranked scheme results |
| iitb.govscheme.optimized_bundle | 10 | Sample optimized bundle |
| iitb.govscheme.pipeline_test_bundle | 10 | End-to-end test results |


## ML Model Details

Algorithm: GBTRegressor (Gradient Boosted Trees)
Features: 22 encoded via StringIndexer and VectorAssembler
Training Data: 4879 user-scheme pairs across 8 synthetic profiles
Metrics: RMSE 0.54 and R2 0.98
Scoring: 50 percent keyword relevance + 25 percent benefit normalized + 25 percent base score
Threshold: Only schemes with final score >= 3.0 returned
Anti-keywords: Occupation-specific blockers prevent irrelevant matches


## Supported Languages

Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, English

State-to-language auto-detection: Maharashtra maps to Marathi, Tamil Nadu to Tamil, UP to Hindi, Karnataka to Kannada, West Bengal to Bengali, Kerala to Malayalam, Gujarat to Gujarati, Andhra Pradesh and Telangana to Telugu, Punjab to Punjabi, Odisha to Odia.


Built at IIT Bombay Hackathon. Powered by Databricks and Sarvam AI.

MIT License

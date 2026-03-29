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

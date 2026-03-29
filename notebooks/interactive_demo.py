# Databricks notebook source
# Check available APIs and install packages
%pip install sarvamai openai-whisper googletrans==4.0.0-rc1 gTTS pydub
import base64  # add this if not already there
dbutils.library.restartPython()

# COMMAND ----------

# =============================================================
# LOAD DATA + CORE PIPELINE FUNCTIONS
# =============================================================
import os, re, warnings, base64, io, json
import numpy as np
import pandas as pd
from datetime import datetime
import requests

from pyspark.sql import SparkSession, functions as F
warnings.filterwarnings("ignore")
spark = SparkSession.builder.getOrCreate()

# Load schemes once into Pandas
PDF_SCHEMES = spark.read.table("iitb.govscheme.schemes").toPandas()
print(f"✅ Loaded {len(PDF_SCHEMES)} schemes")

# ---- Profile Parser ----
def parse_user_profile(text):
    text_lower = text.lower().strip()
    profile = {"age":30,"income_lpa":3.0,"occupation":"general","gender":"any",
               "caste":"general","is_rural":0,"state":"all","description":text}
    for pat in [r'(\d{1,3})\s*(?:years?\s*old|yr|yrs|age)',r'age\s*(?:is|:)?\s*(\d{1,3})',r'i\s+am\s+(\d{1,3})']:
        m=re.search(pat,text_lower)
        if m:
            v=int(m.group(1))
            if 1<=v<=120: profile["age"]=v; break
    for pat,mode in [
        (r'(?:income|earn(?:ing)?|salary|stipend)\s*(?:is|of|:)?\s*(?:rs\.?|₹|inr)?\s*([\d,.]+)\s*(?:lpa|lac|lakh|per\s*annum)',"lpa"),
        (r'(?:rs\.?|₹|inr)\s*([\d,.]+)\s*(?:lpa|lac|lakh)',"lpa"),
        (r'([\d,.]+)\s*(?:lpa|lac\s*per\s*annum)',"lpa"),
        (r'(?:income|earn|salary)\s*(?:is|of|:)?\s*(?:rs\.?|₹|inr)?\s*([\d,.]+)\s*(?:per\s*month|monthly|pm)',"monthly"),
        (r'(?:rs\.?|₹|inr)\s*([\d,.]+)\s*(?:per\s*month|monthly|pm)',"monthly"),
        (r'(?:income|earn|salary)\s*(?:is|of|:)?\s*(?:rs\.?|₹|inr)?\s*([\d,.]+)\s*(?:per\s*year|yearly|annual)',"yearly"),
    ]:
        m=re.search(pat,text_lower)
        if m:
            val=float(m.group(1).replace(",",""))
            if mode=="monthly": profile["income_lpa"]=round((val*12)/100000,2)
            elif mode=="yearly": profile["income_lpa"]=round(val/100000,2)
            else: profile["income_lpa"]=val
            break
    occ_map={"farmer":["farmer","agriculture","farming","kisan","krishi","cultivat"],
        "student":["student","college","university","school","study","pursuing","engineering","btech","mtech","phd"],
        "women":["woman","women","female","girl","housewife","mother","widow","pregnant","mahila"],
        "senior_citizen":["senior","elderly","retired","pension","old age"],
        "worker":["worker","labour","labor","daily wage","construction","factory","maid","domestic"],
        "entrepreneur":["entrepreneur","business","startup","self-employed","self employed","shop","enterprise","msme","udyam"],
        "disabled":["disabled","disability","handicap","divyang","blind","deaf"],
        "minority":["minority","muslim","christian","sikh","buddhist","jain","parsi"]}
    for occ,kws in occ_map.items():
        if any(kw in text_lower for kw in kws): profile["occupation"]=occ; break
    if any(w in text_lower for w in ["female","woman","girl","she ","her ","mother","wife","widow","mahila"]): profile["gender"]="female"
    elif any(w in text_lower for w in ["male","man","boy","he ","his ","father","husband"]): profile["gender"]="male"
    for c in ["sc","st","obc","general","ews"]:
        if re.search(r'\b'+c+r'\b',text_lower): profile["caste"]=c; break
    if any(w in text_lower for w in ["rural","village","gram","gaon","panchayat"]): profile["is_rural"]=1
    elif any(w in text_lower for w in ["urban","city","metro","town","municipal"]): profile["is_rural"]=0
    for st in ["andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa","gujarat","haryana",
        "himachal pradesh","jharkhand","karnataka","kerala","madhya pradesh","maharashtra","manipur","meghalaya",
        "mizoram","nagaland","odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura",
        "uttar pradesh","uttarakhand","west bengal","delhi","jammu","kashmir","ladakh"]:
        if st in text_lower: profile["state"]=st; break
    return profile

# ---- Eligibility + Scoring ----
def get_eligible_and_score(profile):
    """Filter eligible schemes + score them using GBT feature importance weights"""
    import pandas as pd
    
    age = profile.get("age")
    income = profile.get("income_lpa")
    occupation = profile.get("occupation", "").lower().strip()
    gender = profile.get("gender", "").lower().strip()
    caste = profile.get("caste", "").lower().strip()
    area = profile.get("area", "").lower().strip()
    state = profile.get("state", "").lower().strip()
    
    pdf = PDF_SCHEMES.copy()
    
    # --- HARD FILTERS (must match) ---
    
    # Age filter
    if age:
        age = int(age)
        pdf = pdf[(pdf["age_min"] <= age) & (pdf["age_max"] >= age)]
    
    # Income filter
    if income:
        income = float(income)
        pdf = pdf[pdf["income_max_lpa"] >= income]
    
    # Gender filter
    if gender and gender not in ["any", ""]:
        pdf = pdf[(pdf["gender_eligible"].str.lower() == gender) | 
                  (pdf["gender_eligible"].str.lower() == "all") |
                  (pdf["gender_eligible"].str.lower() == "any")]
    
    # Caste filter
    if caste and caste not in ["any", "general", ""]:
        pdf = pdf[(pdf["caste_eligible"].str.lower().str.contains(caste, na=False)) |
                  (pdf["caste_eligible"].str.lower() == "all") |
                  (pdf["caste_eligible"].str.lower() == "any")]
    
    # Area filter
    if area and area not in ["any", ""]:
        if area == "rural":
            pdf = pdf[(pdf["is_rural"] == True) | (pdf["is_rural"].isna())]
        elif area == "urban":
            pdf = pdf[(pdf["is_rural"] == False) | (pdf["is_rural"].isna())]
    
    # State filter
    if state and state not in ["any", ""]:
        pdf = pdf[(pdf["state_eligible"].str.lower().str.contains(state, na=False)) |
                  (pdf["state_eligible"].str.lower() == "all") |
                  (pdf["state_eligible"].str.lower() == "any") |
                  (pdf["state_eligible"].str.lower() == "all states") |
                  (pdf["state_eligible"].isna())]
    
    if len(pdf) == 0:
        return None
    
    # --- RELEVANCE FILTER (soft match based on occupation + keywords) ---
    
    # Build keyword sets for each occupation
    occupation_keywords = {
        "farmer": ["farm", "agri", "crop", "kisan", "soil", "irrigation", "seed", 
                    "fertilizer", "cattle", "dairy", "livestock", "harvest", "rural",
                    "land", "water", "fishery", "horticulture", "plantation"],
        "student": ["student", "education", "scholarship", "school", "college", 
                     "university", "study", "learn", "research", "degree", "exam",
                     "merit", "tuition", "fellowship", "academic", "hostel"],
        "worker": ["worker", "labour", "labor", "employment", "wage", "skill",
                    "construction", "factory", "industry", "unorganized", "informal",
                    "training", "apprentice", "livelihood", "job"],
        "business": ["business", "entrepreneur", "enterprise", "startup", "msme",
                      "loan", "credit", "self-employ", "trade", "commerce", "mudra",
                      "manufacturing", "industry", "subsidy"],
        "senior": ["senior", "elder", "pension", "old age", "retirement", "geriatric",
                    "widow", "disability", "social security", "care home"],
        "woman": ["woman", "women", "girl", "female", "maternal", "pregnancy",
                   "mahila", "beti", "lady", "gender", "widow", "domestic"],
        "unemployed": ["unemploy", "job", "skill", "training", "placement",
                        "livelihood", "self-employ", "employment", "youth"],
    }
    
    # Get relevant keywords
    user_keywords = set()
    
    # From occupation
    if occupation:
        for occ_key, keywords in occupation_keywords.items():
            if occ_key in occupation or occupation in occ_key:
                user_keywords.update(keywords)
    
    # Add universal keywords based on profile
    if age and int(age) < 30:
        user_keywords.update(["youth", "young", "skill", "training", "education"])
    if age and int(age) > 55:
        user_keywords.update(["senior", "pension", "elder", "old age", "retirement"])
    if gender == "female":
        user_keywords.update(["woman", "women", "girl", "female", "mahila", "beti"])
    if caste in ["sc", "st", "obc"]:
        user_keywords.update(["backward", "minority", "tribal", "scheduled", caste])
    if area == "rural":
        user_keywords.update(["rural", "village", "gram", "panchayat"])
    
    # Always include general welfare
    user_keywords.update(["welfare", "benefit", "assistance", "insurance", "health",
                           "housing", "subsidy", "pension", "ration"])
    
    # Score relevance based on keyword matching in description, tags, beneficiaries
    def compute_relevance(row):
        text_fields = ""
        for col in ["description", "tags", "target_beneficiaries", "eligibility_text", 
                     "schemeName_full", "category_str"]:
            val = str(row.get(col, "")).lower()
            if val not in ["nan", "not specified", "none", ""]:
                text_fields += " " + val
        
        if not text_fields.strip():
            return 0.0
        
        matches = sum(1 for kw in user_keywords if kw in text_fields)
        relevance = matches / max(len(user_keywords), 1)
        return relevance
    
    pdf["relevance"] = pdf.apply(compute_relevance, axis=1)
    
    # --- FILTER OUT IRRELEVANT SCHEMES ---
    # Must have at least SOME relevance (>0 keyword matches)
    pdf = pdf[pdf["relevance"] > 0.0]
    
    if len(pdf) == 0:
        # If too strict, fall back to top by basic relevance
        pdf = PDF_SCHEMES.copy()
        pdf["relevance"] = pdf.apply(compute_relevance, axis=1)
        pdf = pdf.nlargest(20, "relevance")
    
    # --- SCORING (using GBT feature importance weights) ---
    
    # Occupation match score (exact or partial)
    def occ_match(row):
        scheme_occ = str(row.get("occupation", "")).lower()
        if not occupation or scheme_occ in ["all", "any", ""]:
            return 0.5
        if occupation in scheme_occ or scheme_occ in occupation:
            return 1.0
        return 0.3
    
    pdf["occ_score"] = pdf.apply(occ_match, axis=1)
    
    # Benefit normalized
    max_benefit = pdf["benefit_inr"].max() if pdf["benefit_inr"].max() > 0 else 1
    pdf["benefit_norm"] = pdf["benefit_inr"].fillna(0) / max_benefit
    
    # Description quality
    pdf["desc_len"] = pdf["description"].fillna("").apply(len)
    max_desc = pdf["desc_len"].max() if pdf["desc_len"].max() > 0 else 1
    pdf["desc_norm"] = pdf["desc_len"] / max_desc
    
    # Final score: GBT-weighted combination
    # Weights from feature importance: match_score=0.497, benefit=0.093, desc_length=0.065
    pdf["raw_score"] = (
        0.40 * pdf["relevance"] +      # keyword relevance (most important)
        0.25 * pdf["occ_score"] +       # occupation match
        0.20 * pdf["benefit_norm"] +    # benefit amount
        0.10 * pdf["desc_norm"] +       # description quality
        0.05                             # base score
    )
    
    # Rescale to 1.0 - 5.0
    min_s = pdf["raw_score"].min()
    max_s = pdf["raw_score"].max()
    if max_s > min_s:
        pdf["final_score"] = 1.0 + 4.0 * (pdf["raw_score"] - min_s) / (max_s - min_s)
    else:
        pdf["final_score"] = 3.0
    
    pdf["final_score"] = pdf["final_score"].round(2)
    
    # Sort by score descending
    pdf = pdf.sort_values("final_score", ascending=False)
    
    return pdf

# ---- Optimizer ----
def optimize_bundle(pdf_ranked, max_schemes=10, max_per_cat=3):
    """Greedy diversity-constrained bundle optimizer"""
    if pdf_ranked is None or len(pdf_ranked) == 0:
        return pdf_ranked
    
    selected = []
    cat_count = {}
    
    for _, row in pdf_ranked.iterrows():
        if len(selected) >= max_schemes:
            break
        
        cat = str(row.get("category_str", "Other"))
        if cat in ["nan", "None", ""]:
            cat = "Other"
        
        # Enforce diversity — max per category
        if cat_count.get(cat, 0) >= max_per_cat:
            continue
        
        # Skip if relevance is too low
        if row.get("relevance", 0) < 0.01:
            continue
        
        selected.append(row)
        cat_count[cat] = cat_count.get(cat, 0) + 1
    
    if not selected:
        # Fallback: just take top N by score
        return pdf_ranked.head(max_schemes)
    
    import pandas as pd
    return pd.DataFrame(selected)

print("✅ Core pipeline ready")
# State → Default Language mapping
STATE_LANGUAGE_MAP = {
    "tamil nadu": "tamil", "karnataka": "kannada", "kerala": "malayalam",
    "andhra pradesh": "telugu", "telangana": "telugu", "maharashtra": "marathi",
    "gujarat": "gujarati", "west bengal": "bengali", "punjab": "punjabi",
    "odisha": "odia", "rajasthan": "hindi", "uttar pradesh": "hindi",
    "bihar": "hindi", "madhya pradesh": "hindi", "chhattisgarh": "hindi",
    "jharkhand": "hindi", "haryana": "hindi", "uttarakhand": "hindi",
    "himachal pradesh": "hindi", "delhi": "hindi", "goa": "english",
    "assam": "bengali", "jammu and kashmir": "hindi", "manipur": "english",
    "meghalaya": "english", "mizoram": "english", "nagaland": "english",
    "sikkim": "english", "tripura": "bengali", "arunachal pradesh": "english",
}

def detect_language_from_state(profile):
    """Auto-detect output language based on user's state"""
    state = profile.get("state", "").lower().strip()
    if state and state != "any":
        for st, lang in STATE_LANGUAGE_MAP.items():
            if st in state or state in st:
                return lang
    return None

print("✅ State→Language mapping loaded (" + str(len(STATE_LANGUAGE_MAP)) + " states)")

# COMMAND ----------

# =============================================================
# VOICE + TRANSLATION (Sarvam AI 🇮🇳 + Fallbacks)
# =============================================================

from gtts import gTTS

# ---- Sarvam AI Key ----
# Get free key at: https://www.sarvam.ai/
SARVAM_API_KEY = "sk_5am9qmcc_79Ymk1REL0cfPkfxPYyFuG2k"

SUPPORTED_LANGUAGES = {
    "hindi":    {"code":"hi","sarvam":"hi-IN","name":"हिंदी"},
    "tamil":    {"code":"ta","sarvam":"ta-IN","name":"தமிழ்"},
    "telugu":   {"code":"te","sarvam":"te-IN","name":"తెలుగు"},
    "bengali":  {"code":"bn","sarvam":"bn-IN","name":"বাংলা"},
    "marathi":  {"code":"mr","sarvam":"mr-IN","name":"मराठी"},
    "gujarati": {"code":"gu","sarvam":"gu-IN","name":"ગુજરાતી"},
    "kannada":  {"code":"kn","sarvam":"kn-IN","name":"ಕನ್ನಡ"},
    "malayalam":{"code":"ml","sarvam":"ml-IN","name":"മലയാളം"},
    "punjabi":  {"code":"pa","sarvam":"pa-IN","name":"ਪੰਜਾਬੀ"},
    "odia":     {"code":"or","sarvam":"or-IN","name":"ଓଡ଼ିଆ"},
    "english":  {"code":"en","sarvam":"en-IN","name":"English"},
}

# ---- Translation ----
def translate_sarvam(text, source_lang, target_lang):
    if not SARVAM_API_KEY: return None
    try:
        # Sarvam needs codes like "hi-IN", not "hi"
        sarvam_codes = {"hi":"hi-IN","en":"en-IN","ta":"ta-IN","te":"te-IN","bn":"bn-IN",
                        "mr":"mr-IN","gu":"gu-IN","kn":"kn-IN","ml":"ml-IN","pa":"pa-IN",
                        "or":"od-IN","as":"as-IN"}
        src = sarvam_codes.get(source_lang, source_lang)
        tgt = sarvam_codes.get(target_lang, target_lang)
        resp = requests.post("https://api.sarvam.ai/translate",
            json={"input":text,"source_language_code":src,"target_language_code":tgt,"mode":"formal"},
            headers={"API-Subscription-Key":SARVAM_API_KEY,"Content-Type":"application/json"}, timeout=10)
        if resp.status_code == 200: return resp.json().get("translated_text")
        else: print(f"   Sarvam translate status: {resp.status_code}")
    except Exception as e: print(f"   Sarvam error: {e}")
    return None

def translate_google(text, source_lang, target_lang):
    try:
        from googletrans import Translator
        return Translator().translate(text, src=source_lang, dest=target_lang).text
    except Exception as e:
        print(f"   Google translate error: {e}")
    return text

def translate_text(text, source_code, target_code):
    if source_code == target_code: return text
    result = translate_sarvam(text, source_code, target_code)
    if result:
        print(f"   ✅ Translated via Sarvam AI ({source_code}→{target_code})")
        return result
    result = translate_google(text, source_code, target_code)
    print(f"   ✅ Translated via Google ({source_code}→{target_code})")
    return result

# ---- STT (Sarvam) ----
def stt_sarvam(audio_bytes, language_code="hi-IN"):
    if not SARVAM_API_KEY: return None
    try:
        resp = requests.post("https://api.sarvam.ai/speech-to-text",
            headers={"API-Subscription-Key":SARVAM_API_KEY},
            files={"file":("audio.wav", audio_bytes, "audio/wav")},
            data={"language_code":language_code,"model":"saarika:v2"}, timeout=30)
        if resp.status_code == 200: return resp.json().get("transcript")
    except: pass
    return None

# ---- TTS ----
def tts_sarvam(text, language_code="hi-IN"):
    if not SARVAM_API_KEY: return None
    try:
        resp = requests.post("https://api.sarvam.ai/text-to-speech",
            json={"inputs":[text[:500]],"target_language_code":language_code,"speaker":"meera","model":"bulbul:v1"},
            headers={"API-Subscription-Key":SARVAM_API_KEY,"Content-Type":"application/json"}, timeout=30)
        if resp.status_code == 200:
            audio_b64 = resp.json().get("audios",[None])[0]
            if audio_b64: return base64.b64decode(audio_b64)
    except: pass
    return None

def tts_gtts(text, language_code="hi"):
    try:
        tts = gTTS(text=text[:1000], lang=language_code, slow=False)
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"   gTTS error: {e}")
    return None

def text_to_speech(text, language):
    """Sarvam AI TTS using bulbul:v2 model"""
    
    lang_map = {
        "hindi": "hi-IN", "tamil": "ta-IN", "telugu": "te-IN",
        "bengali": "bn-IN", "marathi": "mr-IN", "gujarati": "gu-IN",
        "kannada": "kn-IN", "malayalam": "ml-IN", "punjabi": "pa-IN",
        "odia": "od-IN", "english": "en-IN",
        "hi-IN": "hi-IN", "ta-IN": "ta-IN", "te-IN": "te-IN",
        "bn-IN": "bn-IN", "mr-IN": "mr-IN", "gu-IN": "gu-IN",
        "kn-IN": "kn-IN", "ml-IN": "ml-IN", "pa-IN": "pa-IN",
        "od-IN": "od-IN", "en-IN": "en-IN",
    }
    
    lang_code = lang_map.get(language, "en-IN")
    
    if len(text) > 500:
        text = text[:497] + "..."
    
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    for speaker in ["anushka", "abhilash"]:
        payload = {
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "model": "bulbul:v2"          # ← THIS WAS THE FIX
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "audios" in data and data["audios"]:
                    audio_bytes = base64.b64decode(data["audios"][0])
                    print(f"   ✅ Sarvam TTS OK (speaker={speaker}, model=bulbul:v2, lang={lang_code}, {len(audio_bytes)} bytes)")
                    return audio_bytes, data["audios"][0]
            else:
                print(f"   ⚠️ TTS {resp.status_code} with {speaker}: {resp.text[:150]}")
        except Exception as e:
            print(f"   ⚠️ TTS error with {speaker}: {e}")
    
    print("   ❌ All TTS attempts failed")
    return None, None
def speech_to_text(audio_path, language="hindi"):
    """Sarvam AI STT using saarika:v2.5"""
    lang_map = {
        "hindi": "hi-IN", "tamil": "ta-IN", "telugu": "te-IN",
        "bengali": "bn-IN", "marathi": "mr-IN", "gujarati": "gu-IN",
        "kannada": "kn-IN", "malayalam": "ml-IN", "punjabi": "pa-IN",
        "odia": "od-IN", "english": "en-IN",
    }
    lang_code = lang_map.get(language, "hi-IN")
    
    ext = audio_path.rsplit(".", 1)[-1].lower()
    mime_map = {"webm":"audio/webm","wav":"audio/wav","mp3":"audio/mpeg","m4a":"audio/mp4","ogg":"audio/ogg"}
    mime = mime_map.get(ext, "audio/mpeg")
    
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.split("/")[-1], f, mime)}
            data = {"language_code": lang_code, "model": "saarika:v2.5", "with_timestamps": "false"}
            resp = requests.post(url, files=files, data=data, headers=headers, timeout=30)
        
        if resp.status_code == 200:
            result = resp.json()
            transcript = result.get("transcript", "")
            print("   ✅ Sarvam STT OK (" + lang_code + ", " + str(len(transcript)) + " chars)")
            print("   📝 " + transcript[:100])
            return transcript
        else:
            print("   ❌ STT error: " + str(resp.status_code) + " - " + resp.text[:200])
            return None
    except Exception as e:
        print("   ❌ STT exception: " + str(e))
        return None

print("✅ speech_to_text() loaded (model: saarika:v2.5)")
print("✅ speech_to_text() loaded")
# Quick test
import os
print("✅ speech_to_text() loaded")
audio_files = [f for f in os.listdir("/Volumes/iitb/govscheme/raw_data/") if f.endswith((".mp3",".wav",".m4a",".ogg"))]
if audio_files:
    print(f"📂 Audio files found in Volumes: {audio_files}")
else:
    print("📂 No audio files in Volumes yet — upload .mp3/.wav files for voice input")
print(f"✅ Voice pipeline ready | Sarvam: {'🟢 Active' if SARVAM_API_KEY else '🟡 Using fallbacks'}")
print(f"   Languages: {', '.join(SUPPORTED_LANGUAGES.keys())}")

# COMMAND ----------

def govscheme_full_pipeline(input_text, language, max_schemes=10, gen_audio=True, 
                             input_mode="text", audio_path=None, output_language=None):
    """
    Master pipeline with audio input support
    - input_mode: "text" or "audio"
    - audio_path: path to audio file (when input_mode="audio") 
    - output_language: override output language (None = auto-detect from state or use input language)
    """
    import time
    
    steps = []
    start = time.time()
    is_audio_input = (input_mode == "audio")
    
    # --- Step 0: If audio input, run STT first ---
    if is_audio_input and audio_path:
        steps.append(f"🎤 Processing audio input: {audio_path.split('/')[-1]}")
        transcript = speech_to_text(audio_path, language)
        if transcript:
            input_text = transcript
            steps.append(f"   ✅ Transcribed: {transcript[:80]}...")
        else:
            steps.append("   ❌ STT failed — cannot process audio")
            return {
                "profile": {}, "bundle": None, "summary": None,
                "original_text": "(audio failed)", "english_text": "",
                "regional_output": "", "audio_b64": None,
                "steps": steps, "elapsed": round(time.time() - start, 2),
                "is_audio_input": True, "detected_language": language
            }
    
    # --- Step 1: Translate to English ---
    if language != "english":
        steps.append(f"🔄 Translating {language} → English...")
        english_text = translate_text(input_text, language, "english")
        steps.append(f"   ✅ Translated to English")
    else:
        english_text = input_text
        steps.append("📝 English input — no translation needed")
    
    # --- Step 2: Parse profile ---
    steps.append("🔍 Running scheme pipeline...")
    profile = parse_user_profile(english_text)
    
    # --- Step 2b: Auto-detect output language from state ---
    detected_lang = detect_language_from_state(profile)
    if output_language and output_language != "auto":
        final_output_lang = output_language
        steps.append(f"🌐 Output language: {final_output_lang} (user selected)")
    elif detected_lang:
        final_output_lang = detected_lang
        steps.append(f"🌐 Output language: {final_output_lang} (auto-detected from state: {profile.get('state','')})")
    else:
        final_output_lang = language
        steps.append(f"🌐 Output language: {final_output_lang} (same as input)")
    
    # --- Step 3: Get eligible schemes ---
    pdf_ranked = get_eligible_and_score(profile)
    
    if pdf_ranked is None or len(pdf_ranked) == 0:
        steps.append("   ⚠️ No eligible schemes found")
        return {
            "profile": profile, "bundle": None, "summary": None,
            "original_text": input_text, "english_text": english_text,
            "regional_output": "No schemes found", "audio_b64": None,
            "steps": steps, "elapsed": round(time.time() - start, 2),
            "is_audio_input": is_audio_input, "detected_language": final_output_lang
        }
    
    steps.append(f"   ✅ {len(pdf_ranked)} eligible schemes scored")
    
    # --- Step 4: Optimize bundle ---
    bundle = optimize_bundle(pdf_ranked, max_schemes=max_schemes, max_per_cat=3)
    steps.append(f"   ✅ {len(bundle)} schemes selected in optimized bundle")
    
    # --- Step 5: Summary ---
    total_benefit = int(bundle["benefit_inr"].sum()) if "benefit_inr" in bundle.columns else 0
    n_categories = bundle["category_str"].nunique() if "category_str" in bundle.columns else 0
    avg_score = round(bundle["final_score"].mean(), 2) if "final_score" in bundle.columns else 0
    
    summary = {
        "eligible_count": len(pdf_ranked),
        "selected_count": len(bundle),
        "total_benefit": total_benefit,
        "n_categories": n_categories,
        "avg_score": avg_score
    }
    
    # --- Step 6: Build user-friendly output ---
    def format_benefit(val):
        if val >= 100000:
            return str(round(val/100000, 1)) + " lakh rupees"
        elif val >= 1000:
            return str(int(val/1000)) + " thousand rupees"
        else:
            return str(int(val)) + " rupees"
    
    english_output = "Based on your profile, we found " + str(len(bundle)) + " government schemes you may be eligible for.\n\n"
    for idx, (_, row) in enumerate(bundle.iterrows(), 1):
        name = row.get("schemeName_full", row.get("schemeName", "Scheme"))
        benefit = row.get("benefit_inr", 0)
        desc = row.get("description", "")
        short_desc = ""
        if desc and desc != "Not specified":
            first_sent = desc.split(".")[0].strip()
            if len(first_sent) > 10:
                short_desc = first_sent + "."
        english_output += str(idx) + ". " + str(name)
        if benefit and benefit > 0:
            english_output += " - benefit up to " + format_benefit(benefit)
        english_output += ".\n"
        if short_desc:
            english_output += "   " + short_desc + "\n"
        english_output += "\n"
    english_output += "Total benefit available: up to " + format_benefit(total_benefit) + ". "
    english_output += "You can ask me about any specific scheme for more details."
    
    # --- Step 7: Translate to output language ---
    regional_output = english_output
    if final_output_lang != "english":
        steps.append(f"🔄 Translating output → {final_output_lang}...")
        regional_output = translate_text(english_output, "english", final_output_lang)
        steps.append(f"   ✅ Translated to {final_output_lang}")
    
    # --- Step 8: TTS ---
    audio_b64 = None
    if gen_audio:
        steps.append(f"🔊 Generating audio in {final_output_lang}...")
        tts_text = "Based on your profile, we found " + str(len(bundle)) + " government schemes for you.\n"
        for idx, (_, row) in enumerate(bundle.head(5).iterrows(), 1):
            name = row.get("schemeName_full", row.get("schemeName", "Scheme"))
            tts_text += str(idx) + ". " + str(name) + ".\n"
        tts_text += "You can ask me about any specific scheme for more details."
        
        if final_output_lang != "english":
            tts_regional = translate_text(tts_text, "english", final_output_lang)
        else:
            tts_regional = tts_text
        
        audio_result = text_to_speech(tts_regional, final_output_lang)
        if audio_result and audio_result[0] is not None:
            audio_bytes, audio_b64 = audio_result
            audio_path_out = "/Volumes/iitb/govscheme/raw_data/output_" + final_output_lang + ".mp3"
            with open(audio_path_out, "wb") as f:
                f.write(audio_bytes)
            steps.append(f"   ✅ Audio saved ({len(audio_bytes)} bytes)")
        else:
            steps.append("   ❌ Audio generation failed")
    
    elapsed = round(time.time() - start, 2)
    steps.append(f"⏱️ Total time: {elapsed}s")
    
    return {
        "profile": profile,
        "bundle": bundle,
        "summary": summary,
        "original_text": input_text,
        "english_text": english_text,
        "regional_output": regional_output,
        "audio_b64": audio_b64,
        "steps": steps,
        "elapsed": elapsed,
        "is_audio_input": is_audio_input,
        "detected_language": final_output_lang
    }

# COMMAND ----------

# ============================================================
# Cell 5: Widgets — with audio sample examples
# ============================================================
dbutils.widgets.removeAll()

dbutils.widgets.dropdown("input_mode", "Text", 
    ["Text", "Audio (Voice Sample)"], 
    "1. Input Mode")

dbutils.widgets.text("user_input", 
    "मैं उत्तर प्रदेश का 35 साल का किसान हूँ। सालाना आय 2 लाख। OBC वर्ग, ग्रामीण।", 
    "2. Text Input (for Text mode)")

dbutils.widgets.dropdown("input_language", "hindi", 
    ["hindi","tamil","telugu","bengali","marathi","gujarati","kannada","malayalam","punjabi","odia","english"], 
    "3. Input Language")

dbutils.widgets.dropdown("output_language", "auto", 
    ["auto","hindi","tamil","telugu","bengali","marathi","gujarati","kannada","malayalam","punjabi","odia","english"], 
    "4. Output Language (auto = from state)")

dbutils.widgets.dropdown("max_schemes", "10", 
    [str(i) for i in range(3, 21)], 
    "5. Max Schemes")

dbutils.widgets.dropdown("generate_audio", "Yes", ["Yes", "No"], 
    "6. Audio Output")

dbutils.widgets.dropdown("example", "custom", [
    "custom",
    "--- TEXT EXAMPLES ---",
    "text_hindi_farmer",
    "text_tamil_farmer", 
    "text_telugu_woman",
    "text_english_senior",
    "text_marathi_worker",
    "text_bengali_student",
    "--- VOICE EXAMPLES ---",
    "voice_hindi_farmer",
    "voice_tamil_farmer",
    "voice_marathi_worker",
    "voice_telugu_woman",
    "voice_bengali_student"
], "7. Examples")

import os
audio_dir = "/Volumes/iitb/govscheme/raw_data/"
samples = [f for f in os.listdir(audio_dir) if f.startswith("sample_input_")]
print("🎛️ Widgets ready!")
print("")
print("📝 TEXT MODE: Pick a text example or type in box → Run Cell 6")
print("🎤 AUDIO MODE: Pick a voice example → Run Cell 6")
print("")
if samples:
    print("🎤 Voice samples available: " + ", ".join(samples))
else:
    print("⚠️ No voice samples yet — run Cell 8 first to generate them!")

# COMMAND ----------

# ============================================================
# Cell 6: Process + Display — RE-RUN for each query
# ============================================================
import html as html_module
import os

input_mode = dbutils.widgets.get("input_mode")
input_text = dbutils.widgets.get("user_input")
input_language = dbutils.widgets.get("input_language")
output_language = dbutils.widgets.get("output_language")
max_s = int(dbutils.widgets.get("max_schemes"))
gen_audio = dbutils.widgets.get("generate_audio") == "Yes"
example = dbutils.widgets.get("example")

# --- Example Presets ---
text_examples = {
    "text_hindi_farmer": ("मैं उत्तर प्रदेश का 35 साल का किसान हूँ। मेरी सालाना आय 2 लाख रुपये है। OBC वर्ग, ग्रामीण क्षेत्र।", "hindi"),
    "text_tamil_farmer": ("நான் தமிழ்நாட்டைச் சேர்ந்த 30 வயது விவசாயி. ஆண்டு வருமானம் 1.5 லட்சம். SC பிரிவு, கிராமப்புறம்.", "tamil"),
    "text_telugu_woman": ("నేను తెలంగాణకు చెందిన 28 ఏళ్ల మహిళను. నా వార్షిక ఆదాయం 1.8 లక్షలు. SC వర్గం, గ్రామీణ ప్రాంతం.", "telugu"),
    "text_english_senior": ("I am a 65 year old retired person from Kerala. My annual income is 2 lakh rupees. General category, urban area.", "english"),
    "text_marathi_worker": ("मी बिहारमधील 25 वर्षीय बांधकाम कामगार आहे. मला दरमहा 8000 रुपये मिळतात. ST वर्ग, ग्रामीण भाग.", "marathi"),
    "text_bengali_student": ("আমি পশ্চিমবঙ্গের ২২ বছরের ছাত্রী। বার্ষিক আয় ১ লক্ষ টাকা। SC শ্রেণী, গ্রামীণ এলাকা।", "bengali"),
}

voice_examples = {
    "voice_hindi_farmer": ("hindi", "/Volumes/iitb/govscheme/raw_data/sample_input_hindi.mp3"),
    "voice_tamil_farmer": ("tamil", "/Volumes/iitb/govscheme/raw_data/sample_input_tamil.mp3"),
    "voice_marathi_worker": ("marathi", "/Volumes/iitb/govscheme/raw_data/sample_input_marathi.mp3"),
    "voice_telugu_woman": ("telugu", "/Volumes/iitb/govscheme/raw_data/sample_input_telugu.mp3"),
    "voice_bengali_student": ("bengali", "/Volumes/iitb/govscheme/raw_data/sample_input_bengali.mp3"),
}

# --- Determine mode based on example selection ---
is_audio = False
audio_path = None

if example in text_examples:
    input_text, input_language = text_examples[example]
    is_audio = False

elif example in voice_examples:
    input_language, audio_path = voice_examples[example]
    is_audio = True
    gen_audio = True
    if not os.path.exists(audio_path):
        print("❌ Audio file not found: " + audio_path)
        print("Run Cell 8 first to generate voice samples!")
        print("Then re-run this cell.")
        raise FileNotFoundError("Run Cell 8 first")

elif example == "custom":
    if "Audio" in input_mode:
        is_audio = True
        candidates = [
            "/Volumes/iitb/govscheme/raw_data/recorded_audio.webm",
            "/Volumes/iitb/govscheme/raw_data/recorded_audio.wav",
            "/Volumes/iitb/govscheme/raw_data/recorded_audio.mp3",
            "/Volumes/iitb/govscheme/raw_data/sample_input_" + input_language + ".mp3",
        ]
        for p in candidates:
            if os.path.exists(p):
                audio_path = p
                break
        if not audio_path:
            print("❌ No audio file found for " + input_language)
            print("Available files:")
            for f in os.listdir("/Volumes/iitb/govscheme/raw_data/"):
                if f.endswith((".mp3",".wav",".webm")):
                    print("   " + f)
            raise FileNotFoundError("No audio file")

# --- Print mode info ---
if is_audio:
    print("🎤 VOICE MODE | Language: " + input_language.title() + " | Output: " + output_language)
    print("📂 Audio: " + str(audio_path).split("/")[-1])
else:
    print("⌨️ TEXT MODE | Language: " + input_language.title() + " | Output: " + output_language)
    print("📝 " + input_text[:80] + "...")
print("📊 Max schemes: " + str(max_s) + " | 🔊 Audio output: " + str(gen_audio))
print("=" * 60)

# --- Run pipeline ---
result = govscheme_full_pipeline(
    input_text=input_text if not is_audio else "",
    language=input_language,
    max_schemes=max_s,
    gen_audio=gen_audio,
    input_mode="audio" if is_audio else "text",
    audio_path=audio_path if is_audio else None,
    output_language=output_language if output_language != "auto" else None
)

for s in result["steps"]:
    print(s)

# --- Safe HTML helper ---
def safe(text):
    if text is None:
        return ""
    return html_module.escape(str(text))

profile = result.get("profile", {}) or {}
bundle = result.get("bundle", None)
summary = result.get("summary", None)
is_audio_input = result.get("is_audio_input", False)
final_lang = result.get("detected_language", input_language)
lang_display = final_lang.title()

original_text = result.get("original_text", "") or ""
english_text = result.get("english_text", "") or ""
regional_output = result.get("regional_output", "") or ""

# --- Profile Table ---
profile_rows = ""
profile_map = [
    ("age", "👤 Age"), ("income_lpa", "💰 Income"), ("occupation", "💼 Occupation"),
    ("gender", "👫 Gender"), ("caste", "🏷️ Category"), ("area", "🏘️ Area"), ("state", "📍 State")
]
for key, label in profile_map:
    val = profile.get(key, "Any")
    if val is None or str(val).lower() in ["none", "any", ""]:
        val = "Any"
    if key == "income_lpa" and val != "Any":
        val = "Rs " + str(val) + " LPA"
    else:
        val = str(val).title()
    profile_rows += '<tr><td style="padding:8px 12px;font-weight:600;color:#4a5568;border-bottom:1px solid #f0f0f0;">'
    profile_rows += safe(label) + '</td><td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;">'
    profile_rows += safe(val) + '</td></tr>'

# --- Input Section ---
if is_audio_input:
    audio_filename = str(audio_path).split("/")[-1] if audio_path else "audio"
    input_section = '<div style="background:#fef3c7;border-radius:12px;padding:16px;margin-bottom:12px;border-left:4px solid #f59e0b;">'
    input_section += '<p style="margin:0 0 6px;font-weight:600;color:#92400e;font-size:15px;">'
    input_section += '🎤 Voice Input (' + safe(input_language.title()) + ')</p>'
    input_section += '<p style="margin:0 0 6px;color:#78350f;font-size:13px;">🔉 Audio: ' + safe(audio_filename) + '</p>'
    input_section += '<p style="margin:0;color:#92400e;font-size:15px;line-height:1.6;">'
    input_section += '📝 <b>Transcript:</b> ' + safe(original_text) + '</p></div>'
else:
    input_section = '<div style="background:#eef2ff;border-radius:12px;padding:16px;margin-bottom:12px;border-left:4px solid #6366f1;">'
    input_section += '<p style="margin:0 0 6px;font-weight:600;color:#4338ca;font-size:15px;">'
    input_section += '⌨️ Text Input (' + safe(input_language.title()) + ')</p>'
    input_section += '<p style="margin:0;color:#312e81;font-size:15px;line-height:1.6;">'
    input_section += safe(original_text) + '</p></div>'

# --- English Translation ---
english_section = ""
if input_language != "english" and english_text:
    english_section = '<div style="background:#f8fafc;border-radius:12px;padding:12px 16px;margin-bottom:12px;border-left:4px solid #94a3b8;">'
    english_section += '<p style="margin:0;color:#475569;font-size:14px;line-height:1.5;">'
    english_section += '🔄 <b>English:</b> ' + safe(english_text) + '</p></div>'

# --- Language Badge ---
lang_badge = ""
if output_language == "auto" and final_lang != input_language:
    state_name = safe(str(profile.get("state", "")).title())
    lang_badge = '<div style="background:#dbeafe;border-radius:8px;padding:8px 16px;margin-bottom:12px;display:inline-block;">'
    lang_badge += '<span style="color:#1e40af;font-size:13px;">🌐 Output auto-set to <b>'
    lang_badge += safe(lang_display) + '</b> (state: ' + state_name + ')</span></div>'

# --- Audio Player ---
audio_html = ""
if result.get("audio_b64"):
    autoplay_attr = "autoplay" if is_audio_input else ""
    play_note = "▶️ Auto-playing (voice input)" if is_audio_input else "Click ▶️ to listen"
    audio_html = '<div style="background:#f0fdf4;border-radius:12px;padding:16px;margin:16px 0;text-align:center;">'
    audio_html += '<p style="margin:0 0 10px;font-weight:600;color:#166534;font-size:15px;">'
    audio_html += '🔊 Response in ' + safe(lang_display) + '</p>'
    audio_html += '<audio controls ' + autoplay_attr + ' style="width:100%;max-width:500px;">'
    audio_html += '<source src="data:audio/mp3;base64,' + result["audio_b64"] + '" type="audio/mp3"></audio>'
    audio_html += '<p style="margin:6px 0 0;color:#6b7280;font-size:12px;">' + play_note + '</p></div>'

# --- Summary Stats ---
summary_html = ""
if summary:
    bl = str(round(summary["total_benefit"]/100000, 1))
    summary_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0;">'
    
    summary_html += '<div style="background:#eff6ff;padding:14px;border-radius:10px;text-align:center;">'
    summary_html += '<div style="font-size:24px;font-weight:700;color:#2563eb;">' + str(summary["eligible_count"]) + '</div>'
    summary_html += '<div style="font-size:12px;color:#64748b;">Eligible</div></div>'
    
    summary_html += '<div style="background:#f0fdf4;padding:14px;border-radius:10px;text-align:center;">'
    summary_html += '<div style="font-size:24px;font-weight:700;color:#16a34a;">' + str(summary["selected_count"]) + '</div>'
    summary_html += '<div style="font-size:12px;color:#64748b;">Recommended</div></div>'
    
    summary_html += '<div style="background:#fefce8;padding:14px;border-radius:10px;text-align:center;">'
    summary_html += '<div style="font-size:24px;font-weight:700;color:#ca8a04;">Rs ' + bl + 'L</div>'
    summary_html += '<div style="font-size:12px;color:#64748b;">Total Benefit</div></div>'
    
    summary_html += '<div style="background:#fdf4ff;padding:14px;border-radius:10px;text-align:center;">'
    summary_html += '<div style="font-size:24px;font-weight:700;color:#9333ea;">' + str(summary["n_categories"]) + '</div>'
    summary_html += '<div style="font-size:12px;color:#64748b;">Categories</div></div>'
    
    summary_html += '</div>'

# --- Scheme Cards ---
scheme_cards = ""
if bundle is not None and len(bundle) > 0:
    for idx, (_, row) in enumerate(bundle.iterrows(), 1):
        name = safe(str(row.get("schemeName_full", row.get("schemeName", "Scheme"))))
        ministry = str(row.get("ministry_full", row.get("ministry", "")))
        benefit = row.get("benefit_inr", 0)
        desc = str(row.get("description", ""))
        cat = str(row.get("category_str", ""))
        app_process = str(row.get("application_process", ""))
        
        short_desc = ""
        if desc and desc not in ["Not specified", "nan", "None", ""]:
            sents = desc.split(".")
            short_desc = ". ".join(sents[:2]).strip()
            if short_desc and not short_desc.endswith("."):
                short_desc += "."
        
        short_app = ""
        if app_process and app_process not in ["Not specified", "nan", "None", ""]:
            short_app = app_process.split(".")[0].strip()
            if short_app and not short_app.endswith("."):
                short_app += "."
        
        ben_display = ""
        if benefit and float(benefit) > 0:
            if float(benefit) >= 100000:
                ben_display = "Rs " + str(round(float(benefit)/100000, 1)) + " Lakh"
            else:
                ben_display = "Rs " + str(int(benefit))
        
        cat_colors = {"Education":"#3b82f6","Agriculture":"#22c55e","Health":"#ef4444",
                      "Employment":"#f59e0b","Social Welfare":"#8b5cf6","Housing":"#ec4899",
                      "Finance":"#06b6d4","Science":"#6366f1","Women":"#f43f5e","Rural":"#84cc16"}
        fc = cat.split(",")[0].strip() if cat else ""
        bc = cat_colors.get(fc, "#6b7280")
        
        card = '<div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px;'
        card += 'border-left:4px solid ' + bc + ';box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
        
        card += '<div style="display:flex;justify-content:space-between;align-items:start;flex-wrap:wrap;gap:8px;">'
        card += '<h3 style="margin:0;color:#1e293b;font-size:16px;flex:1;">' + str(idx) + '. ' + name + '</h3>'
        if ben_display:
            card += '<span style="background:#ecfdf5;color:#059669;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;white-space:nowrap;">' + safe(ben_display) + '</span>'
        card += '</div>'
        
        if ministry and ministry not in ["nan", "None", ""]:
            card += '<p style="margin:4px 0 8px;color:#64748b;font-size:13px;">🏛️ ' + safe(ministry) + '</p>'
        
        if cat and cat not in ["nan", "None", ""]:
            card += '<span style="display:inline-block;background:' + bc + '20;color:' + bc
            card += ';padding:3px 10px;border-radius:12px;font-size:12px;margin-bottom:8px;">' + safe(cat) + '</span>'
        
        if short_desc:
            card += '<p style="margin:8px 0;color:#374151;font-size:14px;line-height:1.6;">' + safe(short_desc) + '</p>'
        
        if short_app and len(short_app) > 5:
            card += '<p style="margin:6px 0 0;color:#6b7280;font-size:13px;">📋 <b>How to apply:</b> ' + safe(short_app) + '</p>'
        
        card += '</div>'
        scheme_cards += card
else:
    scheme_cards = '<div style="background:#fef2f2;border-radius:12px;padding:24px;text-align:center;">'
    scheme_cards += '<p style="color:#dc2626;font-size:16px;margin:0;">No eligible schemes found</p>'
    scheme_cards += '<p style="color:#9ca3af;font-size:13px;margin:8px 0 0;">Try a different profile</p></div>'

# --- Regional Output ---
regional_html = ""
if regional_output and final_lang != "english":
    regional_html = '<div style="background:#fefce8;border-radius:12px;padding:16px;margin:16px 0;border-left:4px solid #eab308;">'
    regional_html += '<p style="margin:0 0 8px;font-weight:600;color:#854d0e;font-size:15px;">'
    regional_html += '🗣️ Output in ' + safe(lang_display) + '</p>'
    regional_html += '<p style="margin:0;color:#713f12;font-size:14px;line-height:1.7;white-space:pre-line;">'
    regional_html += safe(regional_output[:1500]) + '</p></div>'

# --- Assemble Page ---
page = '<div style="font-family:Segoe UI,system-ui,sans-serif;max-width:920px;margin:0 auto;padding:16px;">'

# Header
page += '<div style="text-align:center;margin-bottom:24px;">'
page += '<h1 style="margin:0;font-size:28px;">🏛️ GovScheme AI</h1>'
page += '<p style="color:#64748b;margin:4px 0 0;font-size:14px;">AI-powered Government Scheme Assistant</p>'
page += '<p style="color:#94a3b8;margin:2px 0 0;font-size:12px;">Sarvam AI + Databricks + MLflow</p>'
page += '</div>'

# Input + Translation + Language badge + Audio
page += input_section
page += english_section
page += lang_badge
page += audio_html

# Two column layout: Profile + Schemes
page += '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px;">'

# Left: Profile
page += '<div style="flex:0 0 230px;">'
page += '<div style="background:white;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1);position:sticky;top:10px;">'
page += '<h3 style="margin:0 0 12px;color:#1e293b;font-size:16px;">👤 Your Profile</h3>'
page += '<table style="width:100%;font-size:14px;border-collapse:collapse;">'
page += profile_rows
page += '</table></div></div>'

# Right: Schemes
page += '<div style="flex:1;min-width:300px;">'
page += '<h2 style="margin:0 0 8px;color:#1e293b;font-size:20px;">✅ Recommended Schemes</h2>'
page += summary_html
page += scheme_cards
page += '</div>'

page += '</div>'

# Regional output
page += regional_html

# Footer
page += '<p style="text-align:center;color:#cbd5e1;font-size:11px;margin-top:24px;">'
page += 'GovScheme AI | Databricks + Sarvam AI + MLflow | IIT Bombay Hackathon</p>'

page += '</div>'

displayHTML(page)

# COMMAND ----------

# ============================================================
# Cell 8: Generate sample voice inputs using Sarvam TTS
# Then test STT round-trip to prove voice pipeline works
# ============================================================

import os

sample_inputs = {
    "hindi": "मैं उत्तर प्रदेश का 35 साल का किसान हूँ। मेरी सालाना आय 2 लाख रुपये है। OBC वर्ग, ग्रामीण क्षेत्र।",
    "tamil": "நான் தமிழ்நாட்டைச் சேர்ந்த 30 வயது விவசாயி. ஆண்டு வருமானம் 1.5 லட்சம். SC பிரிவு, கிராமப்புறம்.",
    "marathi": "मी बिहारमधील 25 वर्षीय बांधकाम कामगार आहे. दरमहा 8000 रुपये मिळतात. ST वर्ग, ग्रामीण भाग.",
    "telugu": "నేను తెలంగాణకు చెందిన 28 ఏళ్ల మహిళను. వార్షిక ఆదాయం 1.8 లక్షలు. SC వర్గం, గ్రామీణ ప్రాంతం.",
    "bengali": "আমি পশ্চিমবঙ্গের 22 বছরের ছাত্রী। বার্ষিক আয় 1 লক্ষ টাকা। SC শ্রেণী, গ্রামীণ এলাকা।"
}

print("🎙️ STEP 1: Generating voice inputs using Sarvam TTS")
print("=" * 60)

generated = []
for lang, text in sample_inputs.items():
    print("\n🔊 " + lang.upper() + ": " + text[:50] + "...")
    audio_result = text_to_speech(text, lang)
    if audio_result and audio_result[0] is not None:
        audio_bytes = audio_result[0]
        path = "/Volumes/iitb/govscheme/raw_data/sample_input_" + lang + ".mp3"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        size_kb = round(len(audio_bytes) / 1024, 1)
        print("   ✅ Saved: " + path + " (" + str(size_kb) + " KB)")
        generated.append((lang, path, text))
    else:
        print("   ❌ TTS failed for " + lang)

print("\n" + "=" * 60)
print("✅ Generated " + str(len(generated)) + " sample audio files!\n")

# --- STEP 2: Test STT on each to verify round-trip ---
print("🎤 STEP 2: Testing STT on each sample (voice → text)")
print("=" * 60)

stt_results = []
for lang, path, original in generated:
    print("\n🎤 STT " + lang.upper() + ":")
    print("   Original: " + original[:60] + "...")
    transcript = speech_to_text(path, lang)
    if transcript:
        print("   Got back: " + transcript[:60] + "...")
        stt_results.append((lang, path, original, transcript))
    else:
        print("   ❌ STT failed")

print("\n" + "=" * 60)
print("✅ STT Results Summary:")
print("=" * 60)
for lang, path, original, transcript in stt_results:
    print("\n🌐 " + lang.upper())
    print("   Input text:  " + original[:70])
    print("   STT output:  " + transcript[:70])

# Save best working language for quick demo
if stt_results:
    best = stt_results[0]
    print("\n" + "=" * 60)
    print("🚀 READY FOR DEMO!")
    print("=" * 60)
    print("Audio files saved. To run voice demo:")
    print("  1. Cell 5: Set Input Mode = 'Record Audio'")
    print("  2. Cell 5: Set Input Language = '" + best[0] + "'")
    print("  3. Run Cell 6")
    print("\nAvailable languages:")
    for lang, path, _, _ in stt_results:
        print("   🎤 " + lang + " → " + path)

# COMMAND ----------



# COMMAND ----------

import requests

audio_path = "/Volumes/iitb/govscheme/raw_data/sample_input_hindi.mp3"

# Check file exists and size
import os
if os.path.exists(audio_path):
    size = os.path.getsize(audio_path)
    print("✅ File exists: " + str(size) + " bytes")
else:
    print("❌ File not found! Run Cell 8 first")

# Test STT
url = "https://api.sarvam.ai/speech-to-text"
headers = {"api-subscription-key": SARVAM_API_KEY}

with open(audio_path, "rb") as f:
    files = {"file": ("test.mp3", f, "audio/mpeg")}
    data = {
        "language_code": "hi-IN",
        "model": "saarika:v2",
        "with_timestamps": "false"
    }
    resp = requests.post(url, files=files, data=data, headers=headers, timeout=30)

print("Status: " + str(resp.status_code))
print("Response: " + resp.text[:500])

# COMMAND ----------

with open("/Volumes/iitb/govscheme/raw_data/sample_input_hindi.mp3", "rb") as f:
    resp = requests.post("https://api.sarvam.ai/speech-to-text",
        files={"file": ("test.mp3", f, "audio/mpeg")},
        data={"language_code": "hi-IN", "model": "saarika:v2.5", "with_timestamps": "false"},
        headers={"api-subscription-key": SARVAM_API_KEY}, timeout=30)
print("Status: " + str(resp.status_code))
print("Response: " + resp.text[:300])

# COMMAND ----------

import shutil

src = "/Volumes/iitb/govscheme/raw_data/schemes_data.json"
dst = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app/schemes_data.json"

shutil.copy(src, dst)
print("✅ Copied schemes_data.json to app folder")

# COMMAND ----------

# Export the PROCESSED schemes table (with all columns the app needs)
pdf = spark.table("iitb.govscheme.schemes").toPandas()
print("Columns: " + str(list(pdf.columns)))
print("Rows: " + str(len(pdf)))

# Save to Volumes first
pdf.to_json("/Volumes/iitb/govscheme/raw_data/schemes_data.json", orient="records")
print("✅ Saved to Volumes")

# Now copy to app folder
app_folder = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"

with open("/Volumes/iitb/govscheme/raw_data/schemes_data.json", "r") as src:
    data = src.read()

with open(app_folder + "/schemes_data.json", "w") as dst:
    dst.write(data)

print("✅ Copied to app folder (" + str(len(pdf)) + " schemes, " + str(len(data)//1024) + " KB)")

# COMMAND ----------

import os

app_folder = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"

for f in os.listdir(app_folder):
    full = os.path.join(app_folder, f)
    if os.path.isdir(full):
        print("📁 " + f + "/")
        for sf in os.listdir(full):
            size = os.path.getsize(os.path.join(full, sf))
            print("   📄 " + sf + " (" + str(size) + " bytes)")
    else:
        size = os.path.getsize(full)
        print("📄 " + f + " (" + str(size) + " bytes)")

# COMMAND ----------

import os
import shutil

app_folder = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"

# Files/folders to DELETE (old template stuff)
to_delete = [
    "backend",          # old backend folder
    "frontend",         # old React frontend
    "package.json",     # Node.js
    "tsconfig.json",    # TypeScript
    "README",           # template readme
]

for f in to_delete:
    path = os.path.join(app_folder, f)
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("🗑️ Deleted folder: " + f)
        else:
            os.remove(path)
            print("🗑️ Deleted file: " + f)
    else:
        print("⏭️ Not found: " + f)

# Verify final structure
print("\n📂 Final app structure:")
for f in sorted(os.listdir(app_folder)):
    full = os.path.join(app_folder, f)
    if os.path.isdir(full):
        print("  📁 " + f + "/")
        for sf in sorted(os.listdir(full)):
            size = os.path.getsize(os.path.join(full, sf))
            print("    📄 " + sf + " (" + str(size) + " bytes)")
    else:
        size = os.path.getsize(full)
        print("  📄 " + f + " (" + str(size) + " bytes)")

# COMMAND ----------

 import os
import shutil

app_folder = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"

# Files/folders to DELETE (old template stuff)
to_delete = [
    "backend",          # old backend folder
    "frontend",         # old React frontend
    "package.json",     # Node.js
    "tsconfig.json",    # TypeScript
    "README",           # template readme
]

for f in to_delete:
    path = os.path.join(app_folder, f)
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("🗑️ Deleted folder: " + f)
        else:
            os.remove(path)
            print("🗑️ Deleted file: " + f)
    else:
        print("⏭️ Not found: " + f)

# Verify final structure
print("\n📂 Final app structure:")
for f in sorted(os.listdir(app_folder)):
    full = os.path.join(app_folder, f)
    if os.path.isdir(full):
        print("  📁 " + f + "/")
        for sf in sorted(os.listdir(full)):
            size = os.path.getsize(os.path.join(full, sf))
            print("    📄 " + sf + " (" + str(size) + " bytes)")
    else:
        size = os.path.getsize(full)
        print("  📄 " + f + " (" + str(size) + " bytes)")

# COMMAND ----------

import os
import shutil

app_folder = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"

# Files/folders to DELETE (old template stuff)
to_delete = [
    "backend",          # old backend folder
    "frontend",         # old React frontend
    "package.json",     # Node.js
    "tsconfig.json",    # TypeScript
    "README",           # template readme
]

for f in to_delete:
    path = os.path.join(app_folder, f)
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("🗑️ Deleted folder: " + f)
        else:
            os.remove(path)
            print("🗑️ Deleted file: " + f)
    else:
        print("⏭️ Not found: " + f)

# Verify final structure
print("\n📂 Final app structure:")
for f in sorted(os.listdir(app_folder)):
    full = os.path.join(app_folder, f)
    if os.path.isdir(full):
        print("  📁 " + f + "/")
        for sf in sorted(os.listdir(full)):
            size = os.path.getsize(os.path.join(full, sf))
            print("    📄 " + sf + " (" + str(size) + " bytes)")
    else:
        size = os.path.getsize(full)
        print("  📄 " + f + " (" + str(size) + " bytes)")

# COMMAND ----------

print("📂 Final structure:")
for f in sorted(os.listdir(app_folder)):
    full = os.path.join(app_folder, f)
    if os.path.isdir(full):
        print("  📁 " + f + "/")
        for sf in sorted(os.listdir(full)):
            print("    📄 " + sf + " (" + str(os.path.getsize(os.path.join(full,sf))) + " bytes)")
    else:
        print("  📄 " + f + " (" + str(os.path.getsize(full)) + " bytes)")

# COMMAND ----------

# ============================================================
# FIX: Shrink schemes_data.json to under 10MB
# ============================================================
import json
import pandas as pd

# Load from Delta
df = spark.table("iitb.govscheme.schemes").toPandas()
print(f"Original columns: {len(df.columns)}")
print(f"Original rows: {len(df)}")

# --- Step 1: Keep ONLY the columns the app actually uses ---
# Check app.py logic: parse_user_profile, get_eligible_and_score, optimize_bundle, display
KEEP_COLUMNS = [
    # Identity
    'schemeId', 'schemeName', 'schemeName_full', 'schemeSlug',
    # Display
    'ministry_full', 'description', 'benefits_text', 'eligibility_text',
    'application_process', 'documents_required',
    # Filtering / scoring
    'tags', 'target_beneficiaries', 'category_str',
    'income_max_lpa', 'age_min', 'age_max', 'occupation',
    'caste_eligible', 'benefit_inr', 'gender_eligible',
    'is_rural', 'state_eligible'
]

# Only keep columns that actually exist
keep = [c for c in KEEP_COLUMNS if c in df.columns]
dropped = [c for c in df.columns if c not in keep]
print(f"Dropping columns: {dropped}")
df = df[keep]

# --- Step 2: Truncate long text fields to save space ---
TEXT_COLS = ['description', 'benefits_text', 'eligibility_text', 
            'application_process', 'documents_required', 'tags', 'target_beneficiaries']
MAX_TEXT_LEN = 500  # characters — enough for display + keyword matching

for col in TEXT_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(lambda x: x[:MAX_TEXT_LEN] if len(x) > MAX_TEXT_LEN else x)

# --- Step 3: Replace verbose "Not specified" / "nan" with empty string ---
df = df.replace({"Not specified": "", "nan": "", "None": ""})
df = df.fillna("")

# --- Step 4: Convert numeric columns properly ---
NUMERIC_COLS = ['income_max_lpa', 'age_min', 'age_max', 'benefit_inr']
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- Step 5: Export as compact JSON (orient='records', no pretty print) ---
records = df.to_dict(orient='records')

# Write compact JSON (no indent, no extra whitespace)
json_str = json.dumps(records, ensure_ascii=False, separators=(',', ':'))
print(f"Compressed JSON size: {len(json_str) / 1024 / 1024:.2f} MB")

# --- Step 6: If still too big, truncate text further ---
if len(json_str) > 9.5 * 1024 * 1024:  # 9.5MB safety margin
    print("Still too large, truncating text to 300 chars...")
    MAX_TEXT_LEN = 300
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x[:MAX_TEXT_LEN] if len(x) > MAX_TEXT_LEN else x)
    records = df.to_dict(orient='records')
    json_str = json.dumps(records, ensure_ascii=False, separators=(',', ':'))
    print(f"Re-compressed JSON size: {len(json_str) / 1024 / 1024:.2f} MB")

if len(json_str) > 9.5 * 1024 * 1024:
    print("STILL too large, truncating to 200 chars...")
    MAX_TEXT_LEN = 200
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x[:MAX_TEXT_LEN] if len(x) > MAX_TEXT_LEN else x)
    records = df.to_dict(orient='records')
    json_str = json.dumps(records, ensure_ascii=False, separators=(',', ':'))
    print(f"Final JSON size: {len(json_str) / 1024 / 1024:.2f} MB")

# --- Step 7: Write to the app folder ---
APP_DIR = "/Workspace/Users/aniketpol2267@gmail.com/databricks_apps/yojana-ai_2026_03_29-07_17/nodejs-fastapi-hello-world-app"

output_path = f"{APP_DIR}/schemes_data.json"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(json_str)

import os
actual_size = os.path.getsize(output_path)
print(f"\n✅ Written to: {output_path}")
print(f"✅ File size: {actual_size / 1024 / 1024:.2f} MB")
print(f"✅ Under 10MB limit: {actual_size < 10 * 1024 * 1024}")
print(f"✅ Schemes count: {len(records)}")

# COMMAND ----------


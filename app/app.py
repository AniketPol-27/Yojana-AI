
import os, json, time, re, math, traceback, base64, io
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import requests as http_requests

# ── Config ──────────────────────────────────────────────────
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_5am9qmcc_79Ymk1REL0cfPkfxPYyFuG2k")
SARVAM_HEADERS = {"api-subscription-key": SARVAM_API_KEY}

LANG_MAP = {
    "hindi":"hi-IN","tamil":"ta-IN","telugu":"te-IN","bengali":"bn-IN",
    "marathi":"mr-IN","gujarati":"gu-IN","kannada":"kn-IN","malayalam":"ml-IN",
    "punjabi":"pa-IN","odia":"od-IN","english":"en-IN"
}

STATE_LANGUAGE_MAP = {
    "tamil nadu":"tamil","karnataka":"kannada","andhra pradesh":"telugu",
    "telangana":"telugu","maharashtra":"marathi","west bengal":"bengali",
    "gujarat":"gujarati","kerala":"malayalam","punjab":"punjabi",
    "odisha":"odia","uttar pradesh":"hindi","bihar":"hindi",
    "madhya pradesh":"hindi","rajasthan":"hindi","delhi":"hindi",
    "haryana":"hindi","uttarakhand":"hindi","jharkhand":"hindi",
    "chhattisgarh":"hindi","himachal pradesh":"hindi","goa":"english",
    "assam":"bengali","meghalaya":"english","manipur":"english",
    "mizoram":"english","nagaland":"english","tripura":"bengali",
    "sikkim":"english","arunachal pradesh":"english",
    "jammu and kashmir":"hindi","ladakh":"hindi",
    "chandigarh":"hindi","puducherry":"tamil",
    "andaman and nicobar":"english","dadra and nagar haveli":"gujarati",
    "daman and diu":"gujarati","lakshadweep":"malayalam"
}

STATE_ALIASES = {
    "up":"uttar pradesh","mp":"madhya pradesh","hp":"himachal pradesh",
    "ap":"andhra pradesh","wb":"west bengal","tn":"tamil nadu",
    "jk":"jammu and kashmir","uk":"uttarakhand","j&k":"jammu and kashmir",
    "nct":"delhi","nct of delhi":"delhi"
}

ALL_STATE_NAMES = [
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh",
    "goa","gujarat","haryana","himachal pradesh","jharkhand","karnataka",
    "kerala","madhya pradesh","maharashtra","manipur","meghalaya","mizoram",
    "nagaland","odisha","punjab","rajasthan","sikkim","tamil nadu",
    "telangana","tripura","uttar pradesh","uttarakhand","west bengal",
    "delhi","jammu and kashmir","ladakh","chandigarh","puducherry",
    "andaman and nicobar","dadra and nagar haveli","daman and diu",
    "lakshadweep"
]

WOMEN_KW = [
    "women","woman","female","girl","mahila","lady","widow","maternity",
    "beti","stree","nari","pregnant","mother","daughter","sakhi",
    "she ","her ","ladies","girls","widows"
]

# ── Load scheme data ────────────────────────────────────────
schemes_df = None

def load_schemes():
    global schemes_df
    path = os.path.join(os.path.dirname(__file__), "schemes_data.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    schemes_df = pd.DataFrame(data)
    print(f"Loaded {len(schemes_df)} schemes")

# ── Sarvam AI Functions ─────────────────────────────────────
def translate_sarvam(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    src = LANG_MAP.get(source_lang, source_lang)
    tgt = LANG_MAP.get(target_lang, target_lang)
    # Chunk into 900-char pieces (API limit)
    chunks = []
    while len(text) > 900:
        idx = text[:900].rfind('. ')
        if idx == -1:
            idx = text[:900].rfind(' ')
        if idx == -1:
            idx = 900
        chunks.append(text[:idx+1])
        text = text[idx+1:]
    chunks.append(text)
    
    translated = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            r = http_requests.post(
                "https://api.sarvam.ai/translate",
                headers={**SARVAM_HEADERS, "Content-Type":"application/json"},
                json={"input":chunk.strip(),"source_language_code":src,
                      "target_language_code":tgt,"model":"mayura:v1"},
                timeout=15
            )
            if r.status_code == 200:
                translated.append(r.json().get("translated_text", chunk))
            else:
                translated.append(chunk)
        except:
            translated.append(chunk)
    return " ".join(translated)


def text_to_speech(text, language):
    lang_code = LANG_MAP.get(language, "en-IN")
    # TTS has ~500 char limit, chunk if needed
    chunks = []
    remaining = text[:2000]  # hard cap
    while len(remaining) > 480:
        idx = remaining[:480].rfind('. ')
        if idx == -1:
            idx = remaining[:480].rfind(' ')
        if idx == -1:
            idx = 480
        chunks.append(remaining[:idx+1])
        remaining = remaining[idx+1:]
    chunks.append(remaining)
    
    all_audio = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            r = http_requests.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={**SARVAM_HEADERS, "Content-Type":"application/json"},
                json={"inputs":[chunk.strip()],"target_language_code":lang_code,
                      "model":"bulbul:v2","speaker":"anushka"},
                timeout=15
            )
            if r.status_code == 200:
                audios = r.json().get("audios", [])
                if audios:
                    all_audio.append(base64.b64decode(audios[0]))
        except:
            pass
    
    if not all_audio:
        return None
    combined = b"".join(all_audio)
    return base64.b64encode(combined).decode()


def speech_to_text(audio_bytes, language, filename="audio.wav"):
    lang_code = LANG_MAP.get(language, "en-IN")
    try:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"language_code": lang_code, "model": "saarika:v2.5"}
        r = http_requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=SARVAM_HEADERS,
            files=files, data=data, timeout=30
        )
        if r.status_code == 200:
            return r.json().get("transcript", "")
    except:
        pass
    return ""

# ── Profile Parser ──────────────────────────────────────────
def parse_user_profile(text):
    t = text.lower().strip()
    profile = {
        "age": None, "income_lpa": None, "occupation": None,
        "gender": None, "caste": None, "area": None, "state": None
    }
    
    # Age
    am = re.search(r'(\d{1,3})\s*(?:year|yr|sal|age|varsh|old|aged|vayasu|bochhor)', t)
    if am:
        profile["age"] = int(am.group(1))
    else:
        am2 = re.search(r'age\s*(?:is|:)?\s*(\d{1,3})', t)
        if am2:
            profile["age"] = int(am2.group(1))
    
    # Income
    im = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|lpa)', t)
    if im:
        profile["income_lpa"] = float(im.group(1))
    else:
        im2 = re.search(r'(?:rs|₹|income)\s*(\d+)', t)
        if im2:
            val = int(im2.group(1))
            if val > 10000:
                profile["income_lpa"] = val / 100000
    
    # Occupation
    occ_map = {
        "farmer": ["farmer","kisan","farming","krishi","agriculture","agricultural","farm "],
        "student": ["student","vidyarthi","studying","college","school","university","padhai"],
        "worker": ["worker","labour","laborer","labourer","mazdoor","shramik","construction","factory"],
        "business": ["business","vyapari","shop","entrepreneur","self-employed","startup","dukan"],
        "teacher": ["teacher","shikshak","professor","lecturer","teaching"],
        "retired": ["retired","pension","senior citizen","seva nivrutt"],
        "unemployed": ["unemployed","berozgar","jobless","no job","no work","no income"]
    }
    for occ, kws in occ_map.items():
        if any(k in t for k in kws):
            profile["occupation"] = occ
            break
    
    # Gender — comprehensive multi-language
    female_kw = [
        "female","woman","girl","mahila","widow","mother","daughter","wife",
        "she ","her ","sister","beti","ladki","aurat","stree","nari","pregnant",
        "sakhi","mata","devi","amma","behan",
        "i am a girl","i am a woman","i am female","i am a widow",
        "i am a mother","i am a daughter","i am a wife",
        "\u092e\u0939\u093f\u0932\u093e","\u0932\u0921\u0915\u0940",
        "\u0938\u094d\u0924\u094d\u0930\u0940","\u0928\u093e\u0930\u0940",
        "\u0914\u0930\u0924","\u092c\u0947\u091f\u0940",
        "\u0baa\u0bc6\u0ba3","\u0bae\u0b95\u0bb3\u0bbf\u0bb0",
        "\u0c38\u0c4d\u0c24\u0c4d\u0c30\u0c40","\u0c2e\u0c39\u0c3f\u0c33",
        "\u09ae\u09b9\u09bf\u09b2\u09be","\u09ae\u09c7\u09af\u09bc\u09c7",
        "\u0cae\u0cb9\u0cbf\u0cb3\u0cc6","\u0d38\u0d4d\u0d24\u0d4d\u0d30\u0d40",
        "\u0a14\u0a30\u0a24","\u0b2e\u0b39\u0b3f\u0b33\u0b3e"
    ]
    male_kw = [
        "male","man","boy","purush","father","husband","son","he ","his ",
        "brother","ladka","aadmi","pita","pati","beta","bhai",
        "i am a man","i am a boy","i am male","i am a father",
        "\u092a\u0941\u0930\u0941\u0937","\u0932\u0921\u0915\u093e",
        "\u0906\u0926\u092e\u0940"
    ]
    for kw in female_kw:
        if kw in t:
            profile["gender"] = "female"
            break
    if not profile["gender"]:
        for kw in male_kw:
            if kw in t:
                profile["gender"] = "male"
                break
    if not profile["gender"]:
        if profile["occupation"] == "farmer":
            profile["gender"] = "male"
    
    # Caste
    caste_map = {"sc":["scheduled caste"," sc ","sc,","dalit"],
                 "st":["scheduled tribe"," st ","st,","tribal","adivasi"],
                 "obc":["obc","other backward"],
                 "ews":["ews","economically weaker"],
                 "general":["general","gen ","unreserved"]}
    for c, kws in caste_map.items():
        if any(k in t or t.startswith(k) for k in kws):
            profile["caste"] = c
            break
    
    # Area
    if any(k in t for k in ["rural","village","gram","gaon","gramin","dehaat"]):
        profile["area"] = "rural"
    elif any(k in t for k in ["urban","city","shahar","nagar","town","metro"]):
        profile["area"] = "urban"
    
    # State
    for alias, full in STATE_ALIASES.items():
        if alias in t:
            profile["state"] = full
            break
    if not profile["state"]:
        for s in sorted(ALL_STATE_NAMES, key=len, reverse=True):
            if s in t:
                profile["state"] = s
                break
    
    return profile

# ── Eligibility + Scoring ───────────────────────────────────
def get_eligible_and_score(profile):
    df = schemes_df.copy()
    
    # Age filter
    if profile.get("age"):
        age = profile["age"]
        if "age_min" in df.columns:
            df = df[df["age_min"].fillna(0).astype(float) <= age]
        if "age_max" in df.columns:
            df = df[df["age_max"].fillna(120).astype(float) >= age]
    
    # Income filter
    if profile.get("income_lpa"):
        inc = profile["income_lpa"]
        if "income_max_lpa" in df.columns:
            df = df[(df["income_max_lpa"].fillna(100).astype(float) >= inc) | 
                     (df["income_max_lpa"].isna())]
    
    # Caste filter
    if profile.get("caste") and "caste_eligible" in df.columns:
        c = profile["caste"].lower()
        df = df[df["caste_eligible"].fillna("all").str.lower().apply(
            lambda x: c in x or "all" in x or x == "")]
    
    # Area filter
    if profile.get("area") and "is_rural" in df.columns:
        if profile["area"] == "rural":
            df = df[df["is_rural"].fillna(1).astype(float) >= 0]
        elif profile["area"] == "urban":
            df = df[df["is_rural"].fillna(0).astype(float) <= 1]
    
    # STRICT STATE FILTER
    user_state = (profile.get("state") or "").lower().strip()
    if user_state and "state_eligible" in df.columns:
        def state_ok(row):
            se = str(row.get("state_eligible","all")).lower()
            if "all" in se or user_state in se:
                desc = str(row.get("description","")).lower() + " " + \
                       str(row.get("schemeName_full","")).lower() + " " + \
                       str(row.get("ministry_full","")).lower()
                for s in ALL_STATE_NAMES:
                    if s != user_state and s in desc:
                        if f"government of {s}" in desc or f"state of {s}" in desc or f"{s} government" in desc:
                            return False
                return True
            return False
        df = df[df.apply(state_ok, axis=1)]
    
    # STRICT GENDER FILTER
    user_gender = (profile.get("gender") or "").lower()
    if user_gender == "male" and "gender_eligible" in df.columns:
        def gender_ok(row):
            ge = str(row.get("gender_eligible","all")).lower()
            if "female" in ge and "all" not in ge:
                return False
            text = str(row.get("schemeName_full","")).lower() + " " + \
                   str(row.get("description","")).lower() + " " + \
                   str(row.get("tags","")).lower()
            hits = sum(1 for w in WOMEN_KW if w in text)
            if hits >= 2:
                return False
            return True
        df = df[df.apply(gender_ok, axis=1)]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # KEYWORD RELEVANCE SCORING
    occ = (profile.get("occupation") or "").lower()
    occ_keywords = {
        "farmer": ["farm","agri","crop","kisan","irrigation","seed","soil","livestock","cattle",
                    "dairy","fishery","horticulture","fertilizer","harvest","mandi","pradhan mantri fasal"],
        "student": ["education","scholarship","study","school","college","university","tuition",
                     "degree","exam","vidyalakshmi","hostel","merit","fellowship"],
        "worker": ["labour","worker","wage","employment","shram","construction","factory",
                    "esi","epf","minimum wage","industrial","occupational"],
        "business": ["enterprise","msme","startup","business","udyam","mudra","loan",
                      "entrepreneur","self-employ","standup","skill"],
        "retired": ["pension","senior","old age","geriatric","elder","retire","vayo"],
        "unemployed": ["employment","skill","training","job","rozgar","placement","apprentice"]
    }
    
    anti_keywords = {
        "farmer": ["scholarship","internship","matric","college","exam","degree","tuition",
                    "university","hostel","merit","fellow"],
        "student": ["pension","senior citizen","old age","widow","retirement","geriatric"],
        "worker": ["scholarship","college","university","degree","tuition"],
        "retired": ["scholarship","student","college","school","matric","university"]
    }
    
    my_kw = occ_keywords.get(occ, [])
    my_anti = anti_keywords.get(occ, [])
    
    max_benefit = df["benefit_inr"].fillna(0).astype(float).max()
    if max_benefit == 0:
        max_benefit = 1
    
    def score_scheme(row):
        text = str(row.get("schemeName_full","")).lower() + " " + \
               str(row.get("description","")).lower() + " " + \
               str(row.get("tags","")).lower() + " " + \
               str(row.get("target_beneficiaries","")).lower()
        
        # Anti-keyword check
        anti_hits = sum(1 for w in my_anti if w in text)
        if anti_hits >= 2:
            return -1
        
        # Relevance
        kw_hits = sum(1 for w in my_kw if w in text) if my_kw else 0
        relevance = min(kw_hits / max(len(my_kw)*0.3, 1), 1.0) * 10
        
        # Benefit normalized
        b = float(row.get("benefit_inr", 0) or 0)
        benefit_norm = (b / max_benefit) * 10
        
        # Base score
        base = 5.0
        
        score = 0.50 * relevance + 0.25 * benefit_norm + 0.25 * base
        return round(score, 2)
    
    df = df.copy()
    df["final_score"] = df.apply(score_scheme, axis=1)
    df = df[df["final_score"] >= 3.0]
    df = df.sort_values("final_score", ascending=False)
    
    return df

# ── Bundle Optimizer ────────────────────────────────────────
def optimize_bundle(df, max_schemes=10):
    if len(df) == 0:
        return df
    selected = []
    cat_count = {}
    for _, row in df.iterrows():
        cat = str(row.get("category_str", "other"))
        if cat_count.get(cat, 0) >= 3:
            continue
        selected.append(row)
        cat_count[cat] = cat_count.get(cat, 0) + 1
        if len(selected) >= max_schemes:
            break
    return pd.DataFrame(selected)

# ── Main Pipeline ───────────────────────────────────────────
def run_pipeline(text, input_language="english", output_language="auto", 
                 max_schemes=10, is_audio=False):
    steps = []
    t0 = time.time()
    
    # Step 1: Translate to English
    if input_language != "english":
        steps.append({"step":"Translation","detail":f"{input_language} → English","status":"running"})
        english_text = translate_sarvam(text, input_language, "english")
        steps[-1]["status"] = "done"
    else:
        english_text = text
        steps.append({"step":"Translation","detail":"Input already in English","status":"done"})
    
    # Step 2: Parse profile
    steps.append({"step":"Profile Parsing","detail":"NLP extraction","status":"running"})
    profile = parse_user_profile(english_text)
    steps[-1]["status"] = "done"
    steps[-1]["detail"] = json.dumps(profile)
    
    # Auto-detect output language
    if output_language == "auto":
        if profile.get("state"):
            output_language = STATE_LANGUAGE_MAP.get(profile["state"], input_language)
        else:
            output_language = input_language
    if output_language == "auto":
        output_language = "english"
    
    # Step 3: Eligibility + Scoring
    steps.append({"step":"ML Scoring","detail":"Filtering & ranking","status":"running"})
    eligible = get_eligible_and_score(profile)
    steps[-1]["status"] = "done"
    steps[-1]["detail"] = f"{len(eligible)} schemes matched"
    
    # Step 4: Optimize bundle
    steps.append({"step":"Optimization","detail":"Diversity-constrained selection","status":"running"})
    bundle = optimize_bundle(eligible, max_schemes)
    steps[-1]["status"] = "done"
    steps[-1]["detail"] = f"{len(bundle)} schemes selected"
    
    # Build scheme list for response
    scheme_list = []
    for i, (_, row) in enumerate(bundle.iterrows()):
        scheme_list.append({
            "rank": i + 1,
            "name": str(row.get("schemeName_full", "Unknown")),
            "ministry": str(row.get("ministry_full", "")),
            "benefit": str(row.get("benefit_inr", "N/A")),
            "category": str(row.get("category_str", "")),
            "description": str(row.get("description", ""))[:600],
            "how_to_apply": str(row.get("application_process", ""))[:1500],
            "eligibility": str(row.get("eligibility_text", ""))[:1000],
            "benefits_text": str(row.get("benefits_text", ""))[:800],
            "documents": str(row.get("documents_required", ""))[:800],
            "score": float(row.get("final_score", 0)),
            "tags": str(row.get("tags", "")),
        })
    
    # Summary text
    total_benefit = bundle["benefit_inr"].fillna(0).astype(float).sum()
    summary_en = f"Based on your profile, we found {len(bundle)} government schemes for you. "
    if len(bundle) > 0:
        summary_en += f"Top scheme: {scheme_list[0]['name']}. "
    summary_en += f"Total estimated benefits: Rs {total_benefit:,.0f}."
    
    # Step 5: Translate output
    summary_regional = summary_en
    if output_language != "english":
        steps.append({"step":"Output Translation","detail":f"English → {output_language}","status":"running"})
        summary_regional = translate_sarvam(summary_en, "english", output_language)
        steps[-1]["status"] = "done"
    
    # Step 6: TTS
    audio_b64 = None
    steps.append({"step":"Voice Synthesis","detail":"Generating audio","status":"running"})
    try:
        audio_b64 = text_to_speech(summary_regional, output_language)
        steps[-1]["status"] = "done"
    except:
        steps[-1]["status"] = "skipped"
    
    elapsed = round(time.time() - t0, 2)
    
    return {
        "profile": profile,
        "input_text": text,
        "english_text": english_text,
        "schemes": scheme_list,
        "total_benefit": total_benefit,
        "summary_en": summary_en,
        "summary_regional": summary_regional,
        "audio_b64": audio_b64,
        "output_language": output_language,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "num_schemes": len(scheme_list)
    }

# ── FastAPI App ─────────────────────────────────────────────
app = FastAPI(title="Yojana AI", version="2.0")

@app.on_event("startup")
def startup():
    load_schemes()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/api/process_text")
async def process_text(request: Request):
    try:
        body = await request.json()
        text = body.get("text","")
        input_lang = body.get("input_language","english")
        output_lang = body.get("output_language","auto")
        max_s = int(body.get("max_schemes", 10))
        result = run_pipeline(text, input_lang, output_lang, max_s, False)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/api/process_audio")
async def process_audio(
    audio: UploadFile = File(...),
    language: str = Form("hindi"),
    max_schemes: int = Form(10)
):
    try:
        audio_bytes = await audio.read()
        transcript = speech_to_text(audio_bytes, language, audio.filename or "audio.wav")
        if not transcript:
            return JSONResponse({"error":"Could not transcribe audio"}, status_code=400)
        result = run_pipeline(transcript, language, "auto", max_schemes, True)
        result["transcript"] = transcript
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/api/translate")
async def translate_endpoint(request: Request):
    try:
        body = await request.json()
        text = body.get("text","")
        target = body.get("target_language","hindi")
        source = body.get("source_language","english")
        translated = translate_sarvam(text, source, target)
        return JSONResponse({"translated_text": translated})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── NEW: Per-section TTS endpoint ───────────────────────────
@app.post("/api/tts")
async def tts_endpoint(request: Request):
    """Generate TTS audio for any text in any supported language.
    Used by per-scheme Listen buttons on the frontend."""
    try:
        body = await request.json()
        text = body.get("text", "")
        language = body.get("language", "english")
        source_language = body.get("source_language", "english")
        
        if not text.strip():
            return JSONResponse({"error": "No text provided"}, status_code=400)
        
        # If source is English and target is different, translate first
        if source_language == "english" and language != "english":
            text = translate_sarvam(text, "english", language)
        elif source_language != "english" and language != source_language:
            # Translate from source to target via English
            english = translate_sarvam(text, source_language, "english")
            text = translate_sarvam(english, "english", language)
        
        audio_b64 = text_to_speech(text, language)
        if audio_b64:
            return JSONResponse({"audio": audio_b64, "translated_text": text})
        else:
            return JSONResponse({"error": "TTS generation failed"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
def health():
    return {"status": "healthy", "schemes_loaded": len(schemes_df) if schemes_df is not None else 0}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

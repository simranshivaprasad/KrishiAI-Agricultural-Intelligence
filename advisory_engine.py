
"""
advisory_engine.py  (offline, no API key needed)
-------------------------------------------------
Turns soil + weather + satellite (NDVI) data into a plain-language advisory
using transparent agronomic rules. Every recommendation can be traced back to
a number in the input data, which is a feature for a public-good platform:
extension officers can audit the logic.
 
What it does:
  - Knows 5 crops, their growth stages, heat tolerance and preferred soil pH
  - Checks moisture, rain, heat, crop greenness (NDVI), disease risk,
    nitrogen, phosphorus, potassium, pH and organic carbon TOGETHER
  - Ranks the findings and returns the most urgent one as the main action
  - Computes a regenerative score (0-10) from how soil-friendly the advised
    actions are (compost/mulch/scouting score high, chemical-only scores low)
  - Reports confidence honestly (lower when inputs are simulated)
  - Writes the advisory in English, Hindi or Marathi (Bhojpuri uses Hindi)
 
Thresholds follow the Soil Health Card ratings used in India:
  N low < 280 kg/ha, P low < 11 kg/ha, K low < 108 kg/ha, organic carbon low < 0.5 %.
Advice is deliberately general (no invented fertilizer quantities): it points
farmers to their Soil Health Card / local Krishi Vigyan Kendra for exact doses.
"""
 
import math
import os
import json
import requests
 
# ----------------------------------------------------------------------
# GEMINI (Google AI Studio, free tier — no credit card required)
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash"  # current model for new projects as of late 2026 (2.5-flash is now restricted to accounts that used it previously)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
 
GEMINI_SYSTEM_PROMPT = """You are an agricultural advisory assistant for Indian smallholder \
farmers, built for a digital public good platform used across multiple states. \
You are given real sensor/satellite/soil data for one farmer's field and must \
produce a short, practical, and HONEST advisory.
 
Rules:
1. Base every recommendation strictly on the numeric data given. Do not invent data.
2. Always include a confidence level (High/Medium/Low) and a one-line reason for it \
   (e.g. "soil data is from the last Soil Health Card cycle, which may be several \
   months old").
3. Include a "regenerative_score" from 0-10 rating how well the top recommendation \
   supports long-term soil health (organic carbon, crop rotation, reduced chemical \
   input) — not just short-term yield.
4. Keep language simple — Grade 8 reading level, no jargon. Assume the reader is a \
   busy farmer, not an agronomist. Never state exact fertilizer quantities — point \
   the farmer to their Soil Health Card or local Krishi Vigyan Kendra for exact doses.
5. Write every text field in the requested language.
6. Respond ONLY with valid JSON matching this exact schema, nothing else, no markdown \
   fences, no preamble, no explanation outside the JSON:
 
{
  "headline": "one sentence summary of the situation",
  "recommendation": "the single most important action to take, in plain language",
  "reasoning": "1-2 sentences on why, referencing the specific data points",
  "regenerative_score": <int 0-10>,
  "regenerative_note": "one sentence on the long-term soil health angle",
  "confidence": "High | Medium | Low",
  "confidence_reason": "one short sentence",
  "secondary_tips": ["tip 1", "tip 2"]
}
"""
 
# ----------------------------------------------------------------------
# CROP PROFILES
# stages = day at which each of 4 stages ends:
#   0 establishment, 1 vegetative, 2 flowering + grain/fruit formation, 3 maturity
# ----------------------------------------------------------------------
CROPS = {
    "wheat":    {"stages": [21, 60, 100, 999], "max_temp": 33, "ph": (6.0, 7.8), "moist_thr": 10, "disease": False},
    "cotton":   {"stages": [30, 70, 130, 999], "max_temp": 38, "ph": (6.0, 8.0), "moist_thr": 10, "disease": False},
    "paddy":    {"stages": [25, 60, 100, 999], "max_temp": 38, "ph": (5.5, 7.5), "moist_thr": 5,  "disease": False},
    "tomato":   {"stages": [25, 50, 100, 999], "max_temp": 35, "ph": (6.0, 7.0), "moist_thr": 10, "disease": True},
    "potato":   {"stages": [25, 50, 80, 999],  "max_temp": 30, "ph": (5.0, 6.5), "moist_thr": 10, "disease": True},
    "maize":    {"stages": [20, 55, 90, 999],  "max_temp": 35, "ph": (5.8, 7.5), "moist_thr": 10, "disease": False},
    "chickpea": {"stages": [20, 45, 90, 999],  "max_temp": 32, "ph": (6.0, 7.5), "moist_thr": 15, "disease": True},
}
 
CROP_NAMES = {
    "wheat":    {"en": "Wheat",    "hi": "गेहूं",   "mr": "गहू"},
    "cotton":   {"en": "Cotton",   "hi": "कपास",    "mr": "कापूस"},
    "paddy":    {"en": "Paddy",    "hi": "धान",     "mr": "भात"},
    "tomato":   {"en": "Tomato",   "hi": "टमाटर",   "mr": "टोमॅटो"},
    "potato":   {"en": "Potato",   "hi": "आलू",     "mr": "बटाटा"},
    "maize":    {"en": "Maize",    "hi": "मक्का",   "mr": "मका"},
    "chickpea": {"en": "Chickpea", "hi": "चना",     "mr": "हरभरा"},
}
 
LANG_CODE = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Bhojpuri": "hi"}
 
# ----------------------------------------------------------------------
# TEXT TEMPLATES  (en / hi / mr)
# ----------------------------------------------------------------------
T = {
    # --- moisture stress ---
    "moist_head": {
        "en": "{crop} field is under moisture stress ({dev}% below normal).",
        "hi": "{crop} के खेत में नमी की कमी है (सामान्य से {dev}% कम)।",
        "mr": "{crop} च्या शेतात ओलाव्याची कमतरता आहे (सामान्यपेक्षा {dev}% कमी).",
    },
    "moist_act_irrigate": {
        "en": "Give a light irrigation within the next 2 days and cover the soil with crop residue (mulch) to hold moisture.",
        "hi": "अगले 2 दिनों में हल्की सिंचाई करें और नमी बनाए रखने के लिए फसल अवशेष की मल्चिंग करें।",
        "mr": "पुढील 2 दिवसांत हलके पाणी द्या आणि ओलावा टिकवण्यासाठी पिकाच्या अवशेषांचे आच्छादन (मल्चिंग) करा.",
    },
    "moist_act_wait": {
        "en": "About {rain} mm of rain is forecast. Hold irrigation and check soil moisture again after the rain.",
        "hi": "अगले 5 दिनों में लगभग {rain} मिमी बारिश का अनुमान है। सिंचाई रोकें और बारिश के बाद नमी दोबारा जांचें।",
        "mr": "पुढील 5 दिवसांत सुमारे {rain} मिमी पावसाचा अंदाज आहे. पाणी देणे थांबवा आणि पावसानंतर ओलावा पुन्हा तपासा.",
    },
    "moist_why": {
        "en": "Soil moisture is {sm}% against a normal of {nm}%.",
        "hi": "मिट्टी में नमी {sm}% है, जबकि सामान्य {nm}% होती है।",
        "mr": "जमिनीतील ओलावा {sm}% आहे, तर सामान्य पातळी {nm}% आहे.",
    },
    # --- heavy rain ---
    "rain_head": {
        "en": "Heavy rain ({rain} mm) is expected in the next 5 days.",
        "hi": "अगले 5 दिनों में भारी बारिश ({rain} मिमी) का अनुमान है।",
        "mr": "पुढील 5 दिवसांत जोरदार पावसाचा ({rain} मिमी) अंदाज आहे.",
    },
    "rain_act": {
        "en": "Clear drainage channels and delay fertilizer and spraying until after the rain to avoid wash-off.",
        "hi": "जल निकासी की नालियां साफ करें और बहने से बचाने के लिए खाद व छिड़काव बारिश के बाद करें।",
        "mr": "निचरा चर स्वच्छ ठेवा आणि वाहून जाऊ नये म्हणून खत व फवारणी पावसानंतर करा.",
    },
    "rain_why": {
        "en": "The 5-day forecast shows {rain} mm of rain, which can waterlog the field and wash away fertilizer.",
        "hi": "5 दिन के पूर्वानुमान में {rain} मिमी बारिश है, जिससे खेत में पानी भर सकता है और खाद बह सकती है।",
        "mr": "5 दिवसांच्या अंदाजात {rain} मिमी पाऊस आहे, त्यामुळे शेतात पाणी साचू शकते आणि खत वाहून जाऊ शकते.",
    },
    # --- heat ---
    "heat_head": {
        "en": "Temperatures ({temp} °C) are above what {crop} handles comfortably.",
        "hi": "तापमान ({temp} °C) {crop} के लिए सहनीय स्तर से अधिक है।",
        "mr": "तापमान ({temp} °C) {crop} साठी सहन करण्याच्या पातळीपेक्षा जास्त आहे.",
    },
    "heat_act": {
        "en": "Irrigate in the early morning or evening and avoid applying fertilizer during the hottest days.",
        "hi": "सुबह जल्दी या शाम को सिंचाई करें और सबसे गर्म दिनों में खाद न डालें।",
        "mr": "पहाटे किंवा संध्याकाळी पाणी द्या आणि सर्वात उष्ण दिवसांत खत देणे टाळा.",
    },
    "heat_why": {
        "en": "Average forecast temperature is {temp} °C; {crop} does best below about {maxt} °C.",
        "hi": "औसत अनुमानित तापमान {temp} °C है; {crop} के लिए लगभग {maxt} °C से कम तापमान बेहतर रहता है।",
        "mr": "सरासरी अंदाजित तापमान {temp} °C आहे; {crop} साठी सुमारे {maxt} °C पेक्षा कमी तापमान चांगले असते.",
    },
    # --- low crop greenness (NDVI) ---
    "vigor_head": {
        "en": "{crop} looks less green than expected for this growth stage.",
        "hi": "इस अवस्था के हिसाब से {crop} की हरियाली अपेक्षा से कम है।",
        "mr": "या टप्प्यासाठी {crop} अपेक्षेपेक्षा कमी हिरवे दिसत आहे.",
    },
    "vigor_act": {
        "en": "Walk the field to look for pests, disease or yellowing leaves, and use the Diagnose Disease tab on any affected leaf.",
        "hi": "खेत में घूमकर कीट, रोग या पीली पत्तियां देखें, और प्रभावित पत्ती पर 'Diagnose Disease' टैब का उपयोग करें।",
        "mr": "शेतात फिरून कीड, रोग किंवा पिवळी पाने तपासा आणि बाधित पानासाठी 'Diagnose Disease' टॅब वापरा.",
    },
    "vigor_why": {
        "en": "Satellite greenness (NDVI) is {ndvi}, while about {exp} is typical for this stage.",
        "hi": "उपग्रह आधारित हरियाली सूचकांक (NDVI) {ndvi} है, जबकि इस अवस्था में लगभग {exp} सामान्य है।",
        "mr": "उपग्रहावर आधारित हिरवेपणा निर्देशांक (NDVI) {ndvi} आहे, तर या टप्प्यात साधारण {exp} अपेक्षित असतो.",
    },
    # --- fungal disease risk ---
    "dis_head": {
        "en": "Wet, mild weather raises fungal disease risk in {crop}.",
        "hi": "नम और हल्के मौसम से {crop} में फफूंद रोगों का खतरा बढ़ जाता है।",
        "mr": "दमट व सौम्य हवामानामुळे {crop} मध्ये बुरशीजन्य रोगांचा धोका वाढतो.",
    },
    "dis_act": {
        "en": "Check leaves twice a week for spots. If you see any, upload a photo in the Diagnose Disease tab before spraying anything.",
        "hi": "सप्ताह में दो बार पत्तियों पर धब्बे जांचें। धब्बे दिखें तो कुछ भी छिड़कने से पहले 'Diagnose Disease' टैब में फोटो अपलोड करें।",
        "mr": "आठवड्यातून दोनदा पानांवर डाग तपासा. डाग दिसल्यास काहीही फवारण्यापूर्वी 'Diagnose Disease' टॅबमध्ये फोटो अपलोड करा.",
    },
    "dis_why": {
        "en": "Forecast rain of {rain} mm with an average of {temp} °C favours blight-type fungal diseases.",
        "hi": "{rain} मिमी बारिश और औसत {temp} °C तापमान झुलसा जैसे फफूंद रोगों के लिए अनुकूल है।",
        "mr": "{rain} मिमी पाऊस आणि सरासरी {temp} °C तापमान करपा (ब्लाइट) सारख्या बुरशीजन्य रोगांना अनुकूल आहे.",
    },
    # --- nitrogen ---
    "n_head": {
        "en": "Soil nitrogen is low ({n} kg/ha) for {crop}.",
        "hi": "{crop} के लिए मिट्टी में नाइट्रोजन कम है ({n} किग्रा/हेक्टेयर)।",
        "mr": "{crop} साठी जमिनीत नायट्रोजन कमी आहे ({n} किग्रॅ/हेक्टर).",
    },
    "n_act": {
        "en": "Apply nitrogen in split doses rather than all at once, using the amount on your Soil Health Card or your local Krishi Vigyan Kendra's advice.",
        "hi": "नाइट्रोजन एक साथ देने के बजाय कई हिस्सों में दें, मात्रा अपने सॉइल हेल्थ कार्ड या स्थानीय कृषि विज्ञान केंद्र की सलाह के अनुसार रखें।",
        "mr": "नायट्रोजन एकाच वेळी न देता हप्त्यांमध्ये विभागून द्या; प्रमाण तुमच्या मृदा आरोग्य पत्रिकेनुसार किंवा स्थानिक कृषी विज्ञान केंद्राच्या सल्ल्यानुसार ठेवा.",
    },
    "n_why": {
        "en": "The Soil Health Card shows {n} kg/ha, below the usual 280 kg/ha 'medium' level.",
        "hi": "सॉइल हेल्थ कार्ड में {n} किग्रा/हेक्टेयर है, जो सामान्य 'मध्यम' स्तर 280 किग्रा/हेक्टेयर से कम है।",
        "mr": "मृदा आरोग्य पत्रिकेत {n} किग्रॅ/हेक्टर आहे, जे सामान्य 'मध्यम' पातळी 280 किग्रॅ/हेक्टर पेक्षा कमी आहे.",
    },
    # --- organic carbon ---
    "oc_head": {
        "en": "Organic carbon is low ({oc}%), so the soil holds less water and nutrients.",
        "hi": "जैविक कार्बन कम है ({oc}%), इसलिए मिट्टी कम पानी और पोषक तत्व रोक पाती है।",
        "mr": "सेंद्रिय कार्बन कमी आहे ({oc}%), त्यामुळे जमीन कमी पाणी व अन्नद्रव्ये धरून ठेवते.",
    },
    "oc_act": {
        "en": "Add farmyard manure or compost, and leave crop residue on the field instead of burning it.",
        "hi": "गोबर की खाद या कम्पोस्ट डालें और फसल अवशेष जलाने के बजाय खेत में ही रहने दें।",
        "mr": "शेणखत किंवा कंपोस्ट घाला आणि पिकाचे अवशेष जाळण्याऐवजी शेतातच राहू द्या.",
    },
    "oc_why": {
        "en": "Soil organic carbon is {oc}%; 0.5% or more is considered healthy.",
        "hi": "मिट्टी में जैविक कार्बन {oc}% है; 0.5% या अधिक स्वस्थ माना जाता है।",
        "mr": "जमिनीतील सेंद्रिय कार्बन {oc}% आहे; 0.5% किंवा अधिक चांगले मानले जाते.",
    },
    # --- phosphorus ---
    "p_head": {
        "en": "Phosphorus is low ({p} kg/ha).",
        "hi": "फॉस्फोरस कम है ({p} किग्रा/हेक्टेयर)।",
        "mr": "स्फुरद कमी आहे ({p} किग्रॅ/हेक्टर).",
    },
    "p_act": {
        "en": "Plan a phosphate fertilizer as the basal dose at the next sowing, and add compost now to help release the phosphorus already in the soil.",
        "hi": "अगली बुवाई पर बेसल डोज के रूप में फॉस्फेट खाद की योजना बनाएं, और मिट्टी में मौजूद फॉस्फोरस उपलब्ध कराने में मदद के लिए अभी कम्पोस्ट डालें।",
        "mr": "पुढील पेरणीच्या वेळी पायाभूत मात्रा म्हणून स्फुरदयुक्त खताचे नियोजन करा आणि जमिनीतील स्फुरद उपलब्ध होण्यासाठी आता कंपोस्ट घाला.",
    },
    "p_why": {
        "en": "Phosphorus is {p} kg/ha, below the usual 11 kg/ha 'medium' level.",
        "hi": "फॉस्फोरस {p} किग्रा/हेक्टेयर है, जो सामान्य 'मध्यम' स्तर 11 किग्रा/हेक्टेयर से कम है।",
        "mr": "स्फुरद {p} किग्रॅ/हेक्टर आहे, जे सामान्य 'मध्यम' पातळी 11 किग्रॅ/हेक्टर पेक्षा कमी आहे.",
    },
    # --- potassium ---
    "k_head": {
        "en": "Potassium is low ({k} kg/ha).",
        "hi": "पोटाश कम है ({k} किग्रा/हेक्टेयर)।",
        "mr": "पालाश कमी आहे ({k} किग्रॅ/हेक्टर).",
    },
    "k_act": {
        "en": "Apply potash fertilizer as advised on your Soil Health Card; potassium helps crops resist drought and disease.",
        "hi": "सॉइल हेल्थ कार्ड की सलाह के अनुसार पोटाश खाद दें; पोटाश फसल को सूखे और रोगों से लड़ने में मदद करता है।",
        "mr": "मृदा आरोग्य पत्रिकेच्या सल्ल्यानुसार पालाशयुक्त खत द्या; पालाश पिकाला दुष्काळ व रोगांचा सामना करण्यास मदत करते.",
    },
    "k_why": {
        "en": "Potassium is {k} kg/ha, below the usual 108 kg/ha 'medium' level.",
        "hi": "पोटाश {k} किग्रा/हेक्टेयर है, जो सामान्य 'मध्यम' स्तर 108 किग्रा/हेक्टेयर से कम है।",
        "mr": "पालाश {k} किग्रॅ/हेक्टर आहे, जे सामान्य 'मध्यम' पातळी 108 किग्रॅ/हेक्टर पेक्षा कमी आहे.",
    },
    # --- pH ---
    "ph_low_head": {
        "en": "Soil is too acidic for {crop} (pH {ph}).",
        "hi": "{crop} के लिए मिट्टी बहुत अम्लीय है (pH {ph})।",
        "mr": "{crop} साठी जमीन खूप आम्लयुक्त आहे (pH {ph}).",
    },
    "ph_low_act": {
        "en": "Ask your soil testing lab about agricultural lime, and add compost alongside it.",
        "hi": "कृषि चूने के बारे में अपनी मिट्टी जांच प्रयोगशाला से पूछें, और साथ में कम्पोस्ट डालें।",
        "mr": "शेतीसाठी चुन्याबद्दल तुमच्या मृदा तपासणी प्रयोगशाळेला विचारा आणि सोबत कंपोस्ट घाला.",
    },
    "ph_high_head": {
        "en": "Soil is too alkaline for {crop} (pH {ph}).",
        "hi": "{crop} के लिए मिट्टी बहुत क्षारीय है (pH {ph})।",
        "mr": "{crop} साठी जमीन खूप विम्लयुक्त (क्षारयुक्त) आहे (pH {ph}).",
    },
    "ph_high_act": {
        "en": "Add compost or green manure, and ask your soil testing lab whether gypsum is needed.",
        "hi": "कम्पोस्ट या हरी खाद डालें, और अपनी मिट्टी जांच प्रयोगशाला से पूछें कि जिप्सम की जरूरत है या नहीं।",
        "mr": "कंपोस्ट किंवा हिरवळीचे खत घाला आणि जिप्समची गरज आहे का ते मृदा तपासणी प्रयोगशाळेला विचारा.",
    },
    "ph_why": {
        "en": "Soil pH is {ph}; {crop} grows best between {lo} and {hi}.",
        "hi": "मिट्टी का pH {ph} है; {crop} के लिए {lo} से {hi} के बीच सबसे अच्छा रहता है।",
        "mr": "जमिनीचा pH {ph} आहे; {crop} साठी {lo} ते {hi} दरम्यान सर्वोत्तम असते.",
    },
    # --- everything fine ---
    "ok_head": {
        "en": "{crop} field conditions look normal this week.",
        "hi": "इस सप्ताह {crop} के खेत की स्थिति सामान्य दिख रही है।",
        "mr": "या आठवड्यात {crop} च्या शेताची स्थिती सामान्य दिसत आहे.",
    },
    "ok_act": {
        "en": "No urgent action needed. Keep checking the field weekly and note down irrigation and fertilizer dates.",
        "hi": "कोई जरूरी कार्रवाई नहीं है। खेत की साप्ताहिक जांच करते रहें और सिंचाई व खाद की तारीखें लिखकर रखें।",
        "mr": "तातडीची कार्यवाही आवश्यक नाही. दर आठवड्याला शेताची तपासणी करा आणि पाणी व खत दिल्याच्या तारखा लिहून ठेवा.",
    },
    "ok_why": {
        "en": "Moisture, rainfall, temperature and soil nutrients are all within normal ranges.",
        "hi": "नमी, बारिश, तापमान और मिट्टी के पोषक तत्व सभी सामान्य सीमा में हैं।",
        "mr": "ओलावा, पाऊस, तापमान आणि जमिनीतील अन्नद्रव्ये सर्व सामान्य मर्यादेत आहेत.",
    },
    # --- regenerative notes ---
    "regen_low_oc": {
        "en": "Building organic carbon with manure and crop residue is the biggest long-term gain for this field.",
        "hi": "गोबर की खाद और फसल अवशेष से जैविक कार्बन बढ़ाना इस खेत के लिए सबसे बड़ा दीर्घकालिक लाभ है।",
        "mr": "शेणखत आणि पिकाच्या अवशेषांनी सेंद्रिय कार्बन वाढवणे हा या शेतासाठी सर्वात मोठा दीर्घकालीन फायदा आहे.",
    },
    "regen_ok_oc": {
        "en": "Keep soil organic matter healthy by retaining crop residue and rotating crops.",
        "hi": "फसल अवशेष रखकर और फसल चक्र अपनाकर मिट्टी में जैविक पदार्थ बनाए रखें।",
        "mr": "पिकाचे अवशेष ठेवून आणि पीक फेरपालट करून जमिनीतील सेंद्रिय पदार्थ टिकवा.",
    },
    # --- tips ---
    "rot_wheat": {
        "en": "Next season, follow wheat with a pulse such as moong or chickpea to put nitrogen back into the soil.",
        "hi": "अगले सीजन में गेहूं के बाद मूंग या चना जैसी दलहनी फसल लगाएं, इससे मिट्टी में नाइट्रोजन वापस आती है।",
        "mr": "पुढील हंगामात गव्हानंतर मूग किंवा हरभरा यांसारखे कडधान्य घ्या, त्यामुळे जमिनीत नायट्रोजन परत येतो.",
    },
    "rot_cotton": {
        "en": "Intercrop cotton with a pulse like moong or cowpea to protect the soil and add nitrogen.",
        "hi": "मिट्टी की सुरक्षा और नाइट्रोजन के लिए कपास के साथ मूंग या लोबिया जैसी दलहनी फसल अंतर-फसल के रूप में लगाएं।",
        "mr": "जमिनीच्या संरक्षणासाठी आणि नायट्रोजनसाठी कापसात मूग किंवा चवळीसारखे कडधान्य आंतरपीक म्हणून घ्या.",
    },
    "rot_paddy": {
        "en": "After paddy, grow a pulse or mustard instead of leaving the field empty, to rebuild the soil.",
        "hi": "धान के बाद खेत खाली छोड़ने के बजाय दलहन या सरसों लगाएं, इससे मिट्टी की सेहत सुधरती है।",
        "mr": "भातानंतर शेत रिकामे ठेवण्याऐवजी कडधान्य किंवा मोहरी घ्या, त्यामुळे जमिनीचे आरोग्य सुधारते.",
    },
    "rot_tomato": {
        "en": "Rotate tomato with a cereal or legume next season to reduce soil-borne disease.",
        "hi": "मिट्टी से फैलने वाले रोग कम करने के लिए अगले सीजन में टमाटर के बाद अनाज या दलहनी फसल लगाएं।",
        "mr": "जमिनीतून पसरणारे रोग कमी करण्यासाठी पुढील हंगामात टोमॅटोनंतर तृणधान्य किंवा कडधान्य घ्या.",
    },
    "rot_potato": {
        "en": "Rotate potato with a cereal such as maize or wheat to break disease cycles.",
        "hi": "रोग चक्र तोड़ने के लिए आलू के बाद मक्का या गेहूं जैसी अनाज वाली फसल लगाएं।",
        "mr": "रोगचक्र खंडित करण्यासाठी बटाट्यानंतर मका किंवा गव्हासारखे तृणधान्य घ्या.",
    },
    "rot_maize": {
        "en": "Follow maize with a pulse such as chickpea or moong to restore soil nitrogen naturally.",
        "hi": "मिट्टी में नाइट्रोजन प्राकृतिक रूप से वापस लाने के लिए मक्का के बाद चना या मूंग जैसी दलहनी फसल लगाएं।",
        "mr": "जमिनीत नैसर्गिकरित्या नायट्रोजन परत आणण्यासाठी मक्यानंतर हरभरा किंवा मूगसारखे कडधान्य घ्या.",
    },
    "rot_chickpea": {
        "en": "As a pulse, chickpea already adds nitrogen to the soil — follow it with a cereal like wheat or maize to make full use of that benefit.",
        "hi": "चना एक दलहनी फसल है जो पहले से मिट्टी में नाइट्रोजन जोड़ती है — इसके बाद गेहूं या मक्का जैसी अनाज फसल लगाकर इसका पूरा लाभ उठाएं।",
        "mr": "हरभरा हे कडधान्य आधीच जमिनीत नायट्रोजन जोडते — त्याचा पुरेपूर फायदा घेण्यासाठी नंतर गहू किंवा मक्यासारखे तृणधान्य घ्या.",
    },
    "tip_kvk": {
        "en": "Confirm any large fertilizer or pesticide purchase with your local Krishi Vigyan Kendra first.",
        "hi": "कोई भी बड़ी खाद या कीटनाशक खरीदने से पहले अपने स्थानीय कृषि विज्ञान केंद्र से पुष्टि कर लें।",
        "mr": "मोठ्या प्रमाणात खत किंवा कीटकनाशक खरेदी करण्यापूर्वी स्थानिक कृषी विज्ञान केंद्राकडून खात्री करा.",
    },
    # --- confidence explanations ---
    "conf_live": {
        "en": "Live weather and satellite data were used; soil values come from the last Soil Health Card cycle and may be months old.",
        "hi": "लाइव मौसम और उपग्रह डेटा का उपयोग किया गया; मिट्टी के आंकड़े पिछले सॉइल हेल्थ कार्ड चक्र के हैं और कई महीने पुराने हो सकते हैं।",
        "mr": "प्रत्यक्ष हवामान व उपग्रह माहिती वापरली आहे; जमिनीची माहिती मागील मृदा आरोग्य पत्रिका चक्रातील असून काही महिने जुनी असू शकते.",
    },
    "conf_sim": {
        "en": "Weather or satellite inputs are simulated in this demo, and soil values come from the last Soil Health Card cycle, which may be months old.",
        "hi": "इस डेमो में मौसम या उपग्रह डेटा अनुमानित (सिम्युलेटेड) है, और मिट्टी के आंकड़े पिछले सॉइल हेल्थ कार्ड चक्र के हैं जो कई महीने पुराने हो सकते हैं।",
        "mr": "या डेमोमध्ये हवामान किंवा उपग्रह माहिती सिम्युलेटेड आहे आणि जमिनीची माहिती मागील मृदा आरोग्य पत्रिका चक्रातील असून काही महिने जुनी असू शकते.",
    },
}
 
 
def _fmt(key, lang, **kw):
    return T[key][lang].format(**kw)
 
 
def _crop_key(crop: str) -> str:
    c = crop.lower()
    if "rice" in c:
        return "paddy"
    if "gram" in c:  # "chickpea" appears in CROPS already; "gram" is the common local name
        return "chickpea"
    for k in CROPS:
        if k in c:
            return k
    return "wheat"  # sensible default if an unknown crop is passed
 
 
def _stage_index(days: int, stages: list) -> int:
    for i, end_day in enumerate(stages):
        if days <= end_day:
            return i
    return len(stages) - 1
 
 
def _is_simulated(source: str) -> bool:
    return str(source or "").lower().startswith("simulated")
 
 
def _rule_based_advisory(
    crop: str,
    growth_stage_days: int,
    soil: dict,
    weather: dict,
    ndvi_data: dict,
    language: str = "English",
) -> dict:
    """The offline, no-API-key advisory engine. Used directly when no Gemini
    key is configured, and automatically as a fallback if a Gemini call
    fails for any reason (network, quota, bad response) — so the app never
    breaks during a live demo."""
    lang = LANG_CODE.get(language, "en")
    ck = _crop_key(crop)
    prof = CROPS[ck]
    cname = CROP_NAMES[ck][lang]
    stage_idx = _stage_index(growth_stage_days, prof["stages"])
 
    rain = round(float(weather["rain_next_5d_mm"]), 1)
    temp = round(float(weather["avg_temp_c"]), 1)
    sm = ndvi_data["soil_moisture_pct"]
    nm = ndvi_data["normal_moisture_pct"]
    dev = ndvi_data["moisture_deviation_pct"]
    ndvi = ndvi_data["ndvi"]
    expected_ndvi = round(0.85 * math.exp(-((growth_stage_days - 60) ** 2) / (2 * 35 ** 2)), 2)
 
    n, p, k = soil["nitrogen_kg_ha"], soil["phosphorus_kg_ha"], soil["potassium_kg_ha"]
    ph, oc = soil["ph"], soil["organic_carbon_pct"]
    n_i, p_i, k_i = int(round(n)), int(round(p)), int(round(k))
 
    # Each finding: priority (lower = more urgent), text keys, params, regen score (0-10)
    findings = []
 
    def add(prio, head, act, why, regen, **kw):
        findings.append({"prio": prio, "head": head, "act": act, "why": why, "regen": regen, "kw": kw})
 
    # 1. Moisture stress (paddy is more sensitive, so a smaller deficit triggers it)
    if dev <= -prof["moist_thr"]:
        act = "moist_act_wait" if rain >= 25 else "moist_act_irrigate"
        add(1, "moist_head", act, "moist_why", 8 if rain >= 25 else 7,
            crop=cname, dev=abs(dev), rain=rain, sm=sm, nm=nm)
 
    # 2. Heavy rain
    if rain >= 60:
        add(2, "rain_head", "rain_act", "rain_why", 5, rain=rain)
 
    # 3. Heat stress
    if temp > prof["max_temp"]:
        add(3, "heat_head", "heat_act", "heat_why", 6, crop=cname, temp=temp, maxt=prof["max_temp"])
 
    # 4. Crop greenness lower than expected for the stage
    if growth_stage_days <= 100 and ndvi < expected_ndvi - 0.12:
        add(4, "vigor_head", "vigor_act", "vigor_why", 7, crop=cname, ndvi=ndvi, exp=expected_ndvi)
 
    # 5. Fungal disease risk (tomato / potato in wet, mild weather)
    if prof["disease"] and rain >= 30 and 15 <= temp <= 28:
        add(5, "dis_head", "dis_act", "dis_why", 8, crop=cname, rain=rain, temp=temp)
 
    # 6. Nitrogen (no point advising top-dressing once the crop is maturing)
    if n < 280 and stage_idx <= 2:
        add(6, "n_head", "n_act", "n_why", 4, crop=cname, n=n_i)
 
    # 7. Organic carbon
    if oc < 0.5:
        add(7, "oc_head", "oc_act", "oc_why", 9, oc=oc)
 
    # 8. Phosphorus
    if p < 11:
        add(8, "p_head", "p_act", "p_why", 6, p=p_i)
 
    # 9. Potassium
    if k < 108:
        add(9, "k_head", "k_act", "k_why", 5, k=k_i)
 
    # 10. pH outside the crop's preferred range
    lo, hi = prof["ph"]
    if ph < lo:
        add(10, "ph_low_head", "ph_low_act", "ph_why", 6, crop=cname, ph=ph, lo=lo, hi=hi)
    elif ph > hi:
        add(10, "ph_high_head", "ph_high_act", "ph_why", 6, crop=cname, ph=ph, lo=lo, hi=hi)
 
    findings.sort(key=lambda f: f["prio"])
 
    if findings:
        top = findings[0]
        headline = _fmt(top["head"], lang, **top["kw"])
        recommendation = _fmt(top["act"], lang, **top["kw"])
        reasoning = _fmt(top["why"], lang, **top["kw"])
        extras = [_fmt(f["act"], lang, **f["kw"]) for f in findings[1:3]]
        # regenerative score: weighted toward the main action, tempered by the rest
        regen_all = [f["regen"] for f in findings]
        score = round(0.6 * top["regen"] + 0.4 * (sum(regen_all) / len(regen_all)))
    else:
        headline = _fmt("ok_head", lang, crop=cname)
        recommendation = _fmt("ok_act", lang)
        reasoning = _fmt("ok_why", lang)
        extras = []
        score = 8
 
    regen_note = _fmt("regen_low_oc" if oc < 0.5 else "regen_ok_oc", lang)
 
    tips = extras + [_fmt("rot_" + ck, lang), _fmt("tip_kvk", lang)]
 
    sim = _is_simulated(weather.get("source")) or _is_simulated(ndvi_data.get("source"))
    confidence = "Medium"  # kept in English: app.py maps High/Medium/Low to icons
    conf_reason = _fmt("conf_sim" if sim else "conf_live", lang)
    if not sim:
        confidence = "High"
 
    return {
        "headline": headline,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "regenerative_score": max(0, min(10, int(score))),
        "regenerative_note": regen_note,
        "confidence": confidence,
        "confidence_reason": conf_reason,
        "secondary_tips": tips,
    }
 
 
def _call_gemini(crop, growth_stage_days, soil, weather, ndvi_data, language) -> dict:
    """Calls the Gemini API directly via REST (no extra SDK dependency).
    Raises on any failure — the caller (generate_advisory) catches this and
    falls back to the rule-based engine."""
    user_prompt = f"""
Crop: {crop}
Days since sowing: {growth_stage_days}
Target language for the output text fields: {language}
 
SOIL DATA (source: {soil.get('source')}):
- Nitrogen: {soil['nitrogen_kg_ha']} kg/ha
- Phosphorus: {soil['phosphorus_kg_ha']} kg/ha
- Potassium: {soil['potassium_kg_ha']} kg/ha
- pH: {soil['ph']}
- Organic carbon: {soil['organic_carbon_pct']}%
- Soil type: {soil['soil_type']}
 
WEATHER (source: {weather.get('source')}):
- Rainfall expected next 5 days: {weather['rain_next_5d_mm']} mm
- Average temperature: {weather['avg_temp_c']} C
 
SATELLITE / NDVI (source: {ndvi_data.get('source')}):
- NDVI (vegetation health index, 0-1): {ndvi_data['ndvi']}
- Soil moisture: {ndvi_data['soil_moisture_pct']}% (regional normal: {ndvi_data['normal_moisture_pct']}%, deviation: {ndvi_data['moisture_deviation_pct']}%)
 
Generate the advisory JSON now.
"""
    body = {
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
        },
    }
    # Free-tier Gemini occasionally returns 503 (server overloaded) — this is
    # transient, not a real failure, so retry a couple of times with a short
    # pause before giving up and falling back to the rule-based engine.
    import time
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(
                GEMINI_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json=body,
                timeout=20,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            last_error = e
            if resp.status_code == 503 and attempt < 2:
                time.sleep(2)
                continue
            raise
    else:
        raise last_error
 
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
 
    # Minimal shape check so a malformed response falls back cleanly
    required = {"headline", "recommendation", "reasoning", "regenerative_score",
                "regenerative_note", "confidence", "confidence_reason", "secondary_tips"}
    if not required.issubset(parsed):
        raise ValueError(f"Gemini response missing expected fields: {required - set(parsed)}")
    return parsed
 
 
def generate_advisory(
    crop: str,
    growth_stage_days: int,
    soil: dict,
    weather: dict,
    ndvi_data: dict,
    language: str = "English",
) -> dict:
    """Public entry point used by app.py (signature unchanged).
 
    If GEMINI_API_KEY is set, tries a real Gemini-generated advisory first.
    On any failure — no key, network issue, quota exceeded, bad response —
    it silently falls back to the offline rule-based engine, so the app
    always returns a usable, complete advisory no matter what."""
    if GEMINI_API_KEY:
        try:
            return _call_gemini(crop, growth_stage_days, soil, weather, ndvi_data, language)
        except Exception as e:
            print(f"[DEBUG] Gemini call failed, using rule-based fallback: {e}")
            pass  # fall through to rule-based engine below
    else:
        print("[DEBUG] GEMINI_API_KEY is empty/not set — using rule-based engine directly")
 
    return _rule_based_advisory(crop, growth_stage_days, soil, weather, ndvi_data, language)
 

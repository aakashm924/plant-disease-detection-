"""
Enhanced Plant Disease Detection Backend
=========================================
Improvements over original:
 1. Grad-CAM heatmap generation (visual AI explanation)
 2. Severity scoring (Mild / Moderate / Severe)
 3. Season-aware advice
 4. Organic vs chemical treatment split
 5. Multi-image ensemble (accepts up to 3 images, averages predictions)
 6. /stats endpoint for dashboard
 7. Better CORS + error handling
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import io, json, base64
from pathlib import Path
from typing import Optional
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'plant_disease_model.keras'
CLASS_INDEX_PATH = BASE_DIR / 'class_indices.json'
HEATMAP_DIR = BASE_DIR / 'heatmaps'
HEATMAP_DIR.mkdir(exist_ok=True)

# ── Load model ──────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)

# ── Load class labels ────────────────────────────────────────────────────────
if CLASS_INDEX_PATH.exists():
    with open(CLASS_INDEX_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    class_labels = {}
    for k, v in class_indices.items():
        if isinstance(v, int):
            class_labels[v] = k
        elif str(k).isdigit():
            class_labels[int(k)] = str(v)
        else:
            class_labels[int(v)] = str(k)
else:
    class_labels = {}

# ── Disease knowledge base ───────────────────────────────────────────────────
DISEASE_DB = {
    "Apple___Apple_scab": {
        "description": "Fungal infection caused by Venturia inaequalis. Creates dark, scabby lesions on leaves and fruit.",
        "severity_guide": {"mild": "< 10% leaf area affected", "moderate": "10–30% affected", "severe": "> 30% or fruit involved"},
        "organic_treatments": ["Neem oil spray (weekly)", "Sulfur dust (avoid >85°F)", "Baking soda solution (1 tsp/L water)"],
        "chemical_treatments": ["Captan fungicide", "Mancozeb", "Thiophanate-methyl"],
        "precautions": ["Remove fallen leaves promptly", "Prune for air circulation", "Avoid overhead watering", "Sanitize tools between cuts"],
        "season_tip": "Most active in wet spring conditions. Apply preventive sprays before bud break.",
        "spread_risk": "HIGH – spores spread by wind and rain",
    },
    "Apple___Black_rot": {
        "description": "Caused by Botryosphaeria obtusa. Creates target-like lesions on fruit and cankers on branches.",
        "severity_guide": {"mild": "Leaf spots only", "moderate": "Some fruit lesions", "severe": "Cankers + fruit rot"},
        "organic_treatments": ["Copper-based fungicide", "Remove mummified fruit", "Prune infected wood to healthy tissue"],
        "chemical_treatments": ["Captan", "Thiophanate-methyl", "Myclobutanil"],
        "precautions": ["Destroy all pruned material", "Avoid wounding bark", "Maintain tree vigour"],
        "season_tip": "Infects through wounds in summer. Monitor after storms.",
        "spread_risk": "MEDIUM – spreads via rain splash and pruning wounds",
    },
    "Apple___Cedar_apple_rust": {
        "description": "Requires both apple and cedar/juniper hosts. Creates bright orange-yellow spots on leaves.",
        "severity_guide": {"mild": "A few spots per leaf", "moderate": "Multiple leaves affected", "severe": "Defoliation risk"},
        "organic_treatments": ["Remove nearby cedar galls in late winter", "Neem oil", "Sulfur spray at bud break"],
        "chemical_treatments": ["Myclobutanil", "Propiconazole", "Triadimefon"],
        "precautions": ["Plant resistant apple varieties", "Remove cedar/juniper hosts if possible"],
        "season_tip": "Cedar galls release spores in spring. Spray before pink bud stage.",
        "spread_risk": "MEDIUM – wind-borne spores from cedar hosts",
    },
    "Apple___healthy": {
        "description": "Plant appears healthy with no visible disease symptoms.",
        "severity_guide": {},
        "organic_treatments": [],
        "chemical_treatments": [],
        "precautions": ["Regular monthly inspection", "Maintain spacing for airflow", "Water at soil level", "Annual dormant pruning"],
        "season_tip": "Apply preventive copper spray before rainy season as insurance.",
        "spread_risk": "NONE",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "White powdery fungal coating on young leaves and shoots caused by Podosphaera clandestina.",
        "severity_guide": {"mild": "Tips of new growth", "moderate": "Several shoots affected", "severe": "Leaves curled + distorted"},
        "organic_treatments": ["Potassium bicarbonate spray", "Neem oil (weekly)", "Milk solution (1:9 milk:water)"],
        "chemical_treatments": ["Sulfur fungicide (not above 85°F)", "Myclobutanil", "Trifloxystrobin"],
        "precautions": ["Avoid nitrogen over-fertilisation (encourages soft growth)", "Prune dense canopy", "Water early morning"],
        "season_tip": "Worst in warm dry days with cool nights. Watch in late spring.",
        "spread_risk": "HIGH – airborne spores spread rapidly",
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Orange-red pustules on both leaf surfaces, caused by Puccinia sorghi.",
        "severity_guide": {"mild": "< 5% leaf coverage", "moderate": "5–20%", "severe": "> 20% or tassels affected"},
        "organic_treatments": ["Plant resistant hybrids", "Remove volunteer corn", "Crop rotation (3+ years)"],
        "chemical_treatments": ["Azoxystrobin", "Propiconazole", "Trifloxystrobin + propiconazole"],
        "precautions": ["Avoid planting near infected fields", "Monitor from VT (tasseling) stage"],
        "season_tip": "Spreads fastest in cool humid conditions (60–77°F). Scout weekly at tasseling.",
        "spread_risk": "HIGH – wind-dispersed spores from southern overwintering areas",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Long cigar-shaped gray-green lesions caused by Exserohilum turcicum.",
        "severity_guide": {"mild": "Lower leaves only", "moderate": "Up to ear leaf", "severe": "Above ear leaf affected"},
        "organic_treatments": ["Crop rotation (2+ years)", "Resistant varieties", "Deep tillage of infected debris"],
        "chemical_treatments": ["Pyraclostrobin", "Azoxystrobin", "Picoxystrobin"],
        "precautions": ["Avoid overhead irrigation", "Scout from V6 stage", "Apply fungicide at VT if conditions favour disease"],
        "season_tip": "Favoured by moderate temperatures (65–80°F) and high humidity.",
        "spread_risk": "HIGH – conidia dispersed by wind and rain",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Rectangular tan/gray lesions limited by leaf veins. Caused by Cercospora zeae-maydis.",
        "severity_guide": {"mild": "A few lesions on lower leaves", "moderate": "Lesions up to mid canopy", "severe": "Upper canopy severely affected"},
        "organic_treatments": ["Crop rotation", "Tillage to reduce residue", "Resistant hybrids"],
        "chemical_treatments": ["Azoxystrobin + propiconazole", "Trifloxystrobin", "Pyraclostrobin"],
        "precautions": ["No-till fields have higher risk", "Plant later if GLS is historically bad in your region"],
        "season_tip": "Most damaging when conditions favour multiple infection cycles. Spray at tasseling.",
        "spread_risk": "HIGH – airborne spores from infected crop residue",
    },
    "Grape___Black_rot": {
        "description": "Caused by Guignardia bidwellii. Circular brown spots with tiny black dots; fruit shrivels to mummy.",
        "severity_guide": {"mild": "Leaf spots only", "moderate": "Some fruit clusters affected", "severe": "Majority of crop lost"},
        "organic_treatments": ["Remove mummified fruit and infected shoots", "Sulfur fungicide", "Copper hydroxide"],
        "chemical_treatments": ["Myclobutanil", "Mancozeb", "Captan"],
        "precautions": ["Destroy all mummies over winter", "Open canopy through pruning", "Avoid shaded dense growth"],
        "season_tip": "Most critical window: bloom to 4 weeks after. Never skip sprays during wet bloom.",
        "spread_risk": "VERY HIGH – mummies persist and reinfect each season",
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Complex fungal disease causing 'tiger-stripe' on leaves and black lesions on berries. Chronic and progressive.",
        "severity_guide": {"mild": "Mild leaf symptoms, no berries", "moderate": "Some berry blackening", "severe": "Apoplexy (sudden death)"},
        "organic_treatments": ["Wound sealant after pruning", "Sodium arsenite (regulated – check local laws)", "Prune in dry weather only"],
        "chemical_treatments": ["No registered fungicide cures Esca; focus on prevention", "Flusilazole as pruning wound protectant"],
        "precautions": ["Prune large wounds cause highest risk", "Use double pruning technique", "Never leave stubs"],
        "season_tip": "Prune during dry cold weather. Protect cuts immediately. Delay pruning reduces risk.",
        "spread_risk": "MEDIUM – enters through pruning wounds",
    },
    "Grape___healthy": {
        "description": "Vine appears healthy. No visible disease symptoms detected.",
        "severity_guide": {},
        "organic_treatments": [],
        "chemical_treatments": [],
        "precautions": ["Annual dormant pruning for airflow", "Thin fruit clusters for berry size", "Monitor for mealybugs and leafhoppers"],
        "season_tip": "Preventive copper + sulfur spray program keeps most diseases at bay.",
        "spread_risk": "NONE",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Caused by Candidatus Liberibacter asiaticus, spread by Asian citrus psyllid. No cure exists.",
        "severity_guide": {"mild": "Blotchy mottle on a few leaves", "moderate": "Multiple shoots affected, small fruit", "severe": "Whole tree – economic removal"},
        "organic_treatments": ["No cure – focus on psyllid management: kaolin clay spray", "Release Tamarixia radiata (parasitic wasp) for biocontrol"],
        "chemical_treatments": ["Imidacloprid (systemic for psyllid)", "Dimethoate (contact insecticide)", "Remove and destroy infected trees"],
        "precautions": ["Source only certified disease-free nursery stock", "Report to agricultural authorities", "Never move plant material across quarantine zones"],
        "season_tip": "Psyllid populations peak in warm humid weather. Monitor and spray psyllids before flush growth.",
        "spread_risk": "EXTREME – quarantine disease, no recovery once systemic",
    },
    "Peach___Bacterial_spot": {
        "description": "Caused by Xanthomonas arboricola. Water-soaked lesions on leaves, fruit and twigs; leaves drop early.",
        "severity_guide": {"mild": "A few leaf spots", "moderate": "Many leaves affected, some fruit lesions", "severe": "Heavy defoliation + unmarketable fruit"},
        "organic_treatments": ["Copper-based bactericide (early season)", "Remove infected twigs during pruning"],
        "chemical_treatments": ["Fixed copper", "Oxytetracycline (only during bloom)", "Avoid copper late – it may cause fruit russet"],
        "precautions": ["Avoid overhead irrigation", "Plant resistant varieties where available", "Spray protectively before predicted rain"],
        "season_tip": "Infection occurs at temperatures 65–86°F with rain or heavy dew. Critical to spray at petal fall.",
        "spread_risk": "HIGH – bacteria spread by rain splash",
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Caused by Xanthomonas euvesicatoria. Water-soaked leaf spots that turn brown; scabby fruit lesions.",
        "severity_guide": {"mild": "Few leaf spots", "moderate": "Significant leaf drop", "severe": "Defoliation + fruit unmarketable"},
        "organic_treatments": ["Copper hydroxide", "Streptomycin (use sparingly to avoid resistance)", "Remove infected plant debris"],
        "chemical_treatments": ["Copper octanoate", "Acibenzolar-S-methyl (SAR inducer)"],
        "precautions": ["Use certified disease-free seed", "Avoid working with wet plants", "Do not overhead irrigate"],
        "season_tip": "Worst when warm and rainy. Spray copper protectively every 5–7 days during wet periods.",
        "spread_risk": "HIGH – spreads on hands, tools, water",
    },
    "Potato___Early_blight": {
        "description": "Caused by Alternaria solani. Concentric ring ('target') spots on older, lower leaves.",
        "severity_guide": {"mild": "Lower leaf spots only", "moderate": "Moving up the canopy", "severe": "Defoliation before maturity"},
        "organic_treatments": ["Remove and destroy lower infected leaves", "Neem oil spray", "Crop rotation (3 years)"],
        "chemical_treatments": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "precautions": ["Mulch to prevent soil splash", "Water at base only", "Avoid over-fertilising nitrogen"],
        "season_tip": "Worst in hot days + cool nights. Spray at first sign or when conditions favor infection.",
        "spread_risk": "MEDIUM – spreads from lower to upper leaves over weeks",
    },
    "Potato___Late_blight": {
        "description": "Caused by Phytophthora infestans. Water-soaked lesions that rapidly destroy the entire plant. Historic famine pathogen.",
        "severity_guide": {"mild": "A few leaf lesions", "moderate": "Multiple plants affected", "severe": "Whole field — emergency action required"},
        "organic_treatments": ["Copper-based fungicides (preventive only)", "Remove and bury infected plants immediately", "Do NOT compost infected material"],
        "chemical_treatments": ["Mancozeb", "Cymoxanil + mancozeb", "Propamocarb + fluopicolide"],
        "precautions": ["EMERGENCY: remove infected plants NOW", "Do not leave infected tubers in field", "Inform neighbouring farms", "Spray borders first to slow spread"],
        "season_tip": "CRITICAL: spreads explosively in cool wet weather. One wet night can destroy a field. Monitor daily.",
        "spread_risk": "EXTREME – can destroy entire crop in days under ideal conditions",
    },
    "Tomato___Bacterial_spot": {
        "description": "Caused by Xanthomonas spp. Small dark spots with yellow halos on leaves; raised scabby fruit lesions.",
        "severity_guide": {"mild": "Few leaf spots", "moderate": "Significant defoliation starting", "severe": "Heavy defoliation + unmarketable fruit"},
        "organic_treatments": ["Copper hydroxide spray", "Remove infected leaves", "Use disease-free transplants"],
        "chemical_treatments": ["Fixed copper + mancozeb", "Acibenzolar-S-methyl"],
        "precautions": ["Never work with wet plants", "Sanitize stakes and cages between seasons", "Avoid overhead irrigation"],
        "season_tip": "Spreads rapidly in warm rainy periods. Apply copper before predicted rain events.",
        "spread_risk": "HIGH – bacteria spread by rain, hands and tools",
    },
    "Tomato___Early_blight": {
        "description": "Caused by Alternaria solani. Classic concentric ring spots on older lower leaves, leading to defoliation.",
        "severity_guide": {"mild": "Lower leaves only", "moderate": "Moving into mid canopy", "severe": "Severe defoliation, fruit exposed to sun scald"},
        "organic_treatments": ["Remove lower infected leaves", "Neem oil spray", "Copper fungicide"],
        "chemical_treatments": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "precautions": ["Mulch generously to stop soil splash", "Water at drip line only", "Stake plants for airflow"],
        "season_tip": "Worst in hot days (>80°F) with cool nights. Begins at bottom of plant and climbs.",
        "spread_risk": "MEDIUM – spreads slowly from lower to upper leaves",
    },
    "Tomato___Late_blight": {
        "description": "Caused by Phytophthora infestans. Water-soaked oily patches rapidly killing leaves, stems and fruit.",
        "severity_guide": {"mild": "Few lesions on leaves", "moderate": "Multiple leaves and some stems", "severe": "Plant collapsing — immediate action"},
        "organic_treatments": ["Remove infected plants", "Copper-based spray (preventive)", "Destroy all infected debris"],
        "chemical_treatments": ["Mancozeb", "Chlorothalonil", "Metalaxyl + mancozeb"],
        "precautions": ["URGENT: remove infected plant material", "Inspect daily in wet weather", "Do not compost infected material"],
        "season_tip": "Explosive in cool wet nights (50–60°F). A single night can initiate an outbreak.",
        "spread_risk": "EXTREME – most destructive tomato disease",
    },
    "Tomato___Leaf_Mold": {
        "description": "Caused by Passalora fulva. Pale yellow patches on upper leaf surface; olive-brown mold on undersides.",
        "severity_guide": {"mild": "A few patches on older leaves", "moderate": "Widespread on lower/mid canopy", "severe": "Severe defoliation"},
        "organic_treatments": ["Improve greenhouse ventilation", "Copper fungicide", "Remove infected leaves"],
        "chemical_treatments": ["Chlorothalonil", "Mancozeb", "Trifloxystrobin"],
        "precautions": ["Keep relative humidity < 85%", "Space plants for airflow", "Water in mornings only"],
        "season_tip": "Greenhouse tomatoes at highest risk. Increase ventilation and reduce watering frequency.",
        "spread_risk": "MEDIUM – thrives in high humidity enclosed spaces",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Caused by Septoria lycopersici. Small circular spots with dark borders and gray centers with tiny black dots.",
        "severity_guide": {"mild": "Lower leaves only", "moderate": "Up to mid canopy", "severe": "Whole plant defoliation risk"},
        "organic_treatments": ["Remove lower leaves immediately", "Copper fungicide", "Neem oil"],
        "chemical_treatments": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "precautions": ["Mulch to reduce soil splash", "Stake and prune for airflow", "Start spraying at first sign"],
        "season_tip": "Begins on lower leaves after first heavy rains. Most damaging in warm wet summers.",
        "spread_risk": "HIGH – spread by rain splash from soil",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Tetranychus urticae – not a disease but a pest. Causes stippling, bronzing and fine webbing on leaves.",
        "severity_guide": {"mild": "Stippling on a few leaves", "moderate": "Webbing visible on multiple leaves", "severe": "Bronzing and leaf drop"},
        "organic_treatments": ["Strong water spray to dislodge mites", "Neem oil spray every 3 days", "Release predatory mites (Phytoseiulus persimilis)", "Insecticidal soap"],
        "chemical_treatments": ["Abamectin", "Bifenazate", "Etoxazole (note: rotate modes of action)"],
        "precautions": ["Avoid dusty conditions (mites thrive)", "Don't over-fertilise nitrogen", "Increase humidity", "Mites develop resistance quickly – rotate chemicals"],
        "season_tip": "Worst in hot dry conditions. Inspect undersides of leaves closely — hard to spot early.",
        "spread_risk": "HIGH – spreads rapidly in hot dry weather; can develop resistance",
    },
    "Tomato___Target_Spot": {
        "description": "Caused by Corynespora cassiicola. Circular brown lesions with concentric rings on leaves and fruit.",
        "severity_guide": {"mild": "A few lesions on lower leaves", "moderate": "Multiple leaves, some fruit lesions", "severe": "Heavy defoliation"},
        "organic_treatments": ["Remove infected leaves", "Copper fungicide", "Improve airflow"],
        "chemical_treatments": ["Chlorothalonil", "Azoxystrobin", "Boscalid"],
        "precautions": ["Avoid overhead irrigation", "Prune lower leaves for ventilation", "Mulch to prevent soil splash"],
        "season_tip": "Favoured by warm humid conditions. Apply protectant sprays before wet periods.",
        "spread_risk": "MEDIUM – spreads through infected plant debris",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "RNA virus causing mottled yellow-green leaf patterns and leaf distortion. No cure exists.",
        "severity_guide": {"mild": "Mild mottling on a few leaves", "moderate": "Widespread mottling and some distortion", "severe": "Stunted plant, poor fruit set"},
        "organic_treatments": ["Remove and destroy infected plants", "Control aphids and whiteflies (virus vectors)", "Wash hands before handling plants"],
        "chemical_treatments": ["No direct cure; control vector insects: imidacloprid for whiteflies", "Reflective mulch to repel aphids"],
        "precautions": ["Use certified virus-free seed", "Disinfect tools with 10% bleach solution", "Never smoke near plants (TMV on tobacco)"],
        "season_tip": "Spreads through mechanical contact. Do not touch healthy plants after handling infected ones.",
        "spread_risk": "HIGH – spreads via hands, tools, and sap",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Spread exclusively by silverleaf whitefly (Bemisia tabaci). Causes upward leaf curling, yellowing and stunting.",
        "severity_guide": {"mild": "Mild curling on young leaves", "moderate": "Significant stunting and curl", "severe": "No fruit set – economic loss"},
        "organic_treatments": ["Yellow sticky traps", "Reflective mulch", "Release Encarsia formosa (parasitic wasp)", "Neem oil spray"],
        "chemical_treatments": ["Imidacloprid (systemic)", "Pymetrozine", "Spiromesifen"],
        "precautions": ["Remove infected plants to stop whitefly feeding on them", "Screen greenhouse vents", "Use resistant tomato varieties where available"],
        "season_tip": "Whitefly populations peak in hot dry weather. Monitor with yellow sticky traps and act early.",
        "spread_risk": "VERY HIGH – one infected plant is a source for whiteflies to spread to neighbours",
    },
    "Tomato___healthy": {
        "description": "Tomato plant looks healthy with no visible disease symptoms.",
        "severity_guide": {},
        "organic_treatments": [],
        "chemical_treatments": [],
        "precautions": ["Stake for airflow", "Remove suckers for open canopy", "Water at soil level only", "Inspect undersides of leaves weekly"],
        "season_tip": "Preventive copper spray at the start of the wet season provides good protection.",
        "spread_risk": "NONE",
    },
    "Squash___Powdery_mildew": {
        "description": "White powdery fungal coating on leaves caused by Podosphaera xanthii. Reduces photosynthesis.",
        "severity_guide": {"mild": "A few white patches", "moderate": "Half the leaf surface covered", "severe": "Whole plant white – fruit quality lost"},
        "organic_treatments": ["Potassium bicarbonate spray", "Milk solution (30% milk in water)", "Neem oil", "Remove badly affected leaves"],
        "chemical_treatments": ["Sulfur fungicide", "Myclobutanil", "Azoxystrobin"],
        "precautions": ["Ensure 3+ feet spacing between plants", "Avoid nitrogen over-fertilisation", "Water early morning at base only"],
        "season_tip": "Most severe in warm days + cool nights. Spray before you see symptoms if neighbours are affected.",
        "spread_risk": "VERY HIGH – airborne spores spread by wind",
    },
    "Strawberry___Leaf_scorch": {
        "description": "Caused by Diplocarpon earlianum. Purple to red spots that enlarge until the leaf looks scorched.",
        "severity_guide": {"mild": "A few red spots on older leaves", "moderate": "Multiple leaves affected", "severe": "Widespread defoliation"},
        "organic_treatments": ["Remove infected leaves", "Copper-based fungicide", "Improve plant spacing"],
        "chemical_treatments": ["Captan", "Myclobutanil", "Azoxystrobin"],
        "precautions": ["Renovate bed after fruiting (mow leaves)", "Improve drainage", "Avoid overhead irrigation"],
        "season_tip": "Infection occurs in wet conditions in spring and autumn. Renovate beds post-harvest to remove infected foliage.",
        "spread_risk": "MEDIUM – rain-splashed spores from infected leaf debris",
    },
    "Potato___Leaf_Mold": {
        "description": "Caused by Fulvia fulva in high-humidity conditions. Yellow-to-brown patches with fuzzy underside growth.",
        "severity_guide": {"mild": "Lower leaf patches", "moderate": "Spread to mid canopy", "severe": "Severe defoliation"},
        "organic_treatments": ["Copper fungicide", "Remove infected leaves", "Improve ventilation"],
        "chemical_treatments": ["Chlorothalonil", "Mancozeb"],
        "precautions": ["Keep humidity below 85%", "Avoid overhead watering", "Space plants for airflow"],
        "season_tip": "Worst in enclosed or humid environments. Open up canopy and ventilate.",
        "spread_risk": "MEDIUM",
    },
}

# Fallback for any unlisted classes
def get_disease_data(class_name):
    data = DISEASE_DB.get(class_name)
    if data:
        return data
    is_healthy = class_name.endswith('healthy')
    return {
        "description": f"{'Plant appears healthy.' if is_healthy else 'Disease detected – consult a local agricultural extension officer for precise identification.'}",
        "severity_guide": {},
        "organic_treatments": [] if is_healthy else ["Consult local agricultural extension"],
        "chemical_treatments": [],
        "precautions": ["Regular monitoring", "Maintain good air circulation", "Water at soil level"],
        "season_tip": "Keep records of when and where symptoms first appear.",
        "spread_risk": "UNKNOWN",
    }

# ── Severity scoring ─────────────────────────────────────────────────────────
def score_severity(confidence: float, class_name: str) -> dict:
    """Derive a severity label from model confidence + class type."""
    is_healthy = class_name.endswith('healthy')
    if is_healthy:
        return {"level": "Healthy", "score": 0, "color": "green"}

    # Use confidence as proxy for severity certainty
    if confidence >= 0.85:
        return {"level": "Severe", "score": 3, "color": "red"}
    elif confidence >= 0.65:
        return {"level": "Moderate", "score": 2, "color": "orange"}
    else:
        return {"level": "Mild", "score": 1, "color": "yellow"}

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def generate_gradcam(image_bytes: bytes, predicted_class_idx: int) -> Optional[str]:
    """Generate Grad-CAM heatmap and return as base64 PNG string."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert('RGB')

        input_shape = model.input_shape
        h = input_shape[1] or 128
        w = input_shape[2] or 128
        img_resized = img.resize((w, h), Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = tf.constant(np.expand_dims(img_array, axis=0))

        # Find last Conv2D layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break

        if last_conv is None:
            return None

        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, predicted_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to original image size
        orig_w, orig_h = img.size
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        orig_array = np.array(img)
        orig_bgr = cv2.cvtColor(orig_array, cv2.COLOR_RGB2BGR)

        # Blend original + heatmap
        superimposed = cv2.addWeighted(orig_bgr, 0.55, heatmap_colored, 0.45, 0)
        _, buffer = cv2.imencode('.jpg', superimposed)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert('RGB')

    input_shape = model.input_shape
    h = input_shape[1] or 128
    w = input_shape[2] or 128
    img = img.resize((w, h), Image.Resampling.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def display_label(class_name: str) -> str:
    if '___' not in class_name:
        return class_name
    plant, disease = class_name.split('___', 1)
    return f"{plant.replace('_', ' ')} — {disease.replace('_', ' ')}"

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Simple in-memory stats tracker
_stats = {"total_scans": 0, "disease_counts": {}, "healthy_count": 0}

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})

    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        predictions = model.predict(img_array, verbose=0)
        top_indices = np.argsort(predictions[0])[-5:][::-1]

        predicted_idx = top_indices[0]
        class_name = class_labels.get(predicted_idx, "Unknown")
        confidence = float(predictions[0][predicted_idx])

        # Grad-CAM
        gradcam_b64 = generate_gradcam(image_bytes, predicted_idx)

        # Disease data
        disease_data = get_disease_data(class_name)
        severity = score_severity(confidence, class_name)

        # Quality check
        second_conf = float(predictions[0][top_indices[1]]) if len(top_indices) > 1 else 0
        confidence_gap = confidence - second_conf
        is_uncertain = confidence_gap < 0.15 or confidence < 0.60

        # Top predictions
        top_predictions = [
            {
                "label": class_labels.get(idx, "Unknown"),
                "display_label": display_label(class_labels.get(idx, "Unknown")),
                "confidence": float(predictions[0][idx]),
            }
            for idx in top_indices[:3]
        ]

        # Update stats
        _stats["total_scans"] += 1
        if class_name.endswith("healthy"):
            _stats["healthy_count"] += 1
        else:
            _stats["disease_counts"][class_name] = _stats["disease_counts"].get(class_name, 0) + 1

        return jsonify({
            "label": class_name,
            "display_label": display_label(class_name),
            "confidence": confidence,
            "severity": severity,
            "is_healthy": class_name.endswith("healthy"),
            "is_uncertain": is_uncertain,
            "confidence_gap": confidence_gap,
            "disease_info": {
                "description": disease_data["description"],
                "spread_risk": disease_data["spread_risk"],
                "season_tip": disease_data["season_tip"],
                "severity_guide": disease_data["severity_guide"],
            },
            "treatments": {
                "organic": disease_data["organic_treatments"],
                "chemical": disease_data["chemical_treatments"],
            },
            "precautions": disease_data["precautions"],
            "top_predictions": top_predictions,
            "gradcam": gradcam_b64,  # base64 JPEG string
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/stats', methods=['GET'])
def stats():
    top_diseases = sorted(_stats["disease_counts"].items(), key=lambda x: x[1], reverse=True)[:5]
    return jsonify({
        "total_scans": _stats["total_scans"],
        "healthy_count": _stats["healthy_count"],
        "disease_count": _stats["total_scans"] - _stats["healthy_count"],
        "top_diseases": [{"name": display_label(k), "count": v} for k, v in top_diseases],
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'plant_disease_model.keras'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=False, use_reloader=False)

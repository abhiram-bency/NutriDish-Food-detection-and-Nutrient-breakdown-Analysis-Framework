import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("nutridish_model.keras")

CLASSES = [
    "apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
    "beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito",
    "bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake",
    "ceviche","cheese_plate","cheesecake","chicken_curry","chicken_quesadilla",
    "chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder",
    "club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
    "deviled_eggs","donuts","dumplings","edamame","eggs_benedict",
    "escargots","falafel","filet_mignon","fish_and_chips","foie_gras",
    "french_fries","french_onion_soup","french_toast","fried_calamari","fried_rice",
    "frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich",
    "grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup",
    "hot_dog","huevos_rancheros","hummus","ice_cream","lasagna",
    "lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup",
    "mussels","nachos","omelette","onion_rings","oysters",
    "pad_thai","paella","pancakes","panna_cotta","peking_duck",
    "pho","pizza","pork_chop","poutine","prime_rib",
    "pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto",
    "samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits",
    "spaghetti_bolognese","spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake",
    "sushi","tacos","takoyaki","tiramisu","tuna_tartare","waffles",
]

from PIL import Image
import sys

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path:
    print("Usage: python3 test.py path/to/food.jpg")
    exit()

img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))
arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
arr = arr[np.newaxis, ...]

preds = model.predict(arr, verbose=0)[0]
top5 = np.argsort(preds)[::-1][:5]

print("\n🍽️  Top predictions:")
for i, idx in enumerate(top5):
    print(f"  {i+1}. {CLASSES[idx]:<30} {preds[idx]:.1%}")


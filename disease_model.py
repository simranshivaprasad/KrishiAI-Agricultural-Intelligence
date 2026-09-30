"""
disease_model.py
-----------------
Two things live here:

1. train() — fine-tunes a MobileNetV2 (transfer learning) on the PlantVillage
   dataset for crop leaf disease classification. Run this ONCE on Day 1,
   ideally with a GPU (Kaggle/Colab free tier is fine — see README).
   Training from a pretrained ImageNet backbone on PlantVillage typically
   reaches 90%+ val accuracy in under an hour on a single GPU because the
   dataset is clean and well-separated.

2. predict() — loads the saved model and classifies a single uploaded leaf
   image. Used by app.py. Falls back to a stub prediction if no trained
   model file is present yet, so the rest of the app can be built/demoed
   before training finishes.

Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease (PlantVillage)
Download it, unzip into ./plantvillage_data/ with one subfolder per class
(this is the default Kaggle layout), then run:

    python disease_model.py --train --data_dir ./plantvillage_data --epochs 5
"""

import os
import argparse
import json

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "disease_model.keras")
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), "models", "class_indices.json")

# Treatment guide — keys match the exact 15 class folder names from the
# emmarex/plantdisease Kaggle dataset (confirmed from actual training output).
TREATMENT_GUIDE = {
    "Pepper__bell___Bacterial_spot": "Use copper-based bactericide. Avoid working in fields when leaves are wet. Remove and destroy infected plant debris after harvest.",
    "Pepper__bell___healthy": "No action needed. Continue regular monitoring.",
    "Potato___Early_blight": "Apply fungicide at first sign of spots. Practice 2-3 year crop rotation with non-solanaceous crops.",
    "Potato___Late_blight": "Urgent: apply systemic fungicide immediately. Remove volunteer plants. Spreads very fast in cool, wet conditions.",
    "Potato___healthy": "No action needed. Continue regular monitoring.",
    "Tomato_Bacterial_spot": "Use copper-based bactericide. Avoid overhead irrigation and working in wet fields.",
    "Tomato_Early_blight": "Remove infected lower leaves. Apply copper-based fungicide. Mulch to prevent soil splash onto leaves.",
    "Tomato_Late_blight": "Remove and destroy infected plants immediately. Apply fungicide (chlorothalonil/mancozeb). High risk of rapid spread in humid weather.",
    "Tomato_Leaf_Mold": "Improve air circulation and reduce humidity (space plants, prune lower leaves). Apply fungicide if severe.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves promptly. Apply fungicide. Avoid overhead watering and rotate crops.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray with water to dislodge mites, or use insecticidal soap/neem oil. Mites thrive in hot, dry, dusty conditions.",
    "Tomato__Target_Spot": "Remove infected leaves. Apply fungicide. Improve air circulation between plants.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "No cure once infected — remove and destroy affected plants. Control whiteflies (the carrier insect) to prevent spread to healthy plants.",
    "Tomato__Tomato_mosaic_virus": "No cure once infected — remove and destroy affected plants. Wash hands and tools after handling infected plants; avoid tobacco use near plants (can carry a related virus).",
    "Tomato_healthy": "No action needed. Continue regular monitoring.",
}


def train(data_dir: str, epochs: int = 5, img_size: int = 160, batch_size: int = 32):
    """Fine-tune MobileNetV2 on a PlantVillage-style directory
    (one subfolder per class). Saves model + class index mapping."""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=(img_size, img_size), batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=(img_size, img_size), batch_size=batch_size,
    )

    class_names = train_ds.class_names
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f)

    # Preprocessing (MobileNetV2's expected [-1, 1] scaling) is applied here,
    # OUTSIDE the saved model graph, on purpose. Baking preprocess_input's raw
    # TF ops into the model graph makes the saved file version-fragile across
    # TensorFlow/Keras releases (this is what caused the "Unknown layer:
    # TrueDivide" error when loading a model trained on one TF version with a
    # newer one installed locally). Keeping it as a plain tf.data.map() step
    # instead means the saved model itself only contains standard Keras
    # layers, which load reliably everywhere.
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze backbone for fast transfer learning

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"Training on {len(class_names)} classes: {class_names}")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Class mapping saved to {CLASS_MAP_PATH}")


def predict(image_path: str, img_size: int = 160) -> dict:
    """Classify a single leaf image. Returns dict with class, confidence,
    and treatment advice. If no trained model exists yet, returns a clearly
    labeled stub so the rest of the app remains demoable."""
    if not os.path.exists(MODEL_PATH):
        return {
            "predicted_class": "Tomato___Early_blight",
            "confidence": 0.0,
            "treatment": TREATMENT_GUIDE.get("Tomato___Early_blight", "N/A"),
            "note": "STUB RESULT — no trained model found at models/disease_model.keras. "
                    "Run `python disease_model.py --train --data_dir <path>` first.",
        }

    import tensorflow as tf
    import numpy as np

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH) as f:
        class_map = json.load(f)

    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # Same preprocessing used during training (now applied manually here,
    # since it's no longer baked into the saved model — see train()).
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    predicted_class = class_map[str(top_idx)]
    confidence = float(preds[top_idx])

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "treatment": TREATMENT_GUIDE.get(predicted_class, "Consult local agricultural extension officer for treatment guidance."),
        "note": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./plantvillage_data")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.train:
        train(args.data_dir, epochs=args.epochs)
    else:
        print("Pass --train --data_dir <path> to train the model.")

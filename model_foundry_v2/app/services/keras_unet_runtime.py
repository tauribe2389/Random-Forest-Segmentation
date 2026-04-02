"""TensorFlow/Keras runtime helpers for the Keras U-Net model family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in guarded integration paths
        raise RuntimeError(
            "TensorFlow is not installed in the active Python environment. "
            "Use the model_foundry_v2 virtual environment for Keras U-Net execution."
        ) from exc
    return tf


def build_unet_model(input_shape: tuple[int, int, int], num_classes: int, base_filters: int = 16):
    tf = _import_tensorflow()
    layers = tf.keras.layers

    inputs = tf.keras.Input(shape=input_shape)

    c1 = layers.Conv2D(base_filters, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(base_filters, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(base_filters * 2, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(base_filters * 2, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(base_filters * 4, 3, activation="relu", padding="same")(p2)
    b = layers.Conv2D(base_filters * 4, 3, activation="relu", padding="same")(b)

    u2 = layers.UpSampling2D()(b)
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(base_filters * 2, 3, activation="relu", padding="same")(u2)
    c3 = layers.Conv2D(base_filters * 2, 3, activation="relu", padding="same")(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(base_filters, 3, activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(base_filters, 3, activation="relu", padding="same")(c4)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(c4)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mf_v2_unet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def train_unet_from_manifest(training_manifest: dict, output_dir: Path, model_config: dict) -> dict:
    tf = _import_tensorflow()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_height = int(model_config.get("target_height", 64))
    target_width = int(model_config.get("target_width", 64))
    epochs = int(model_config.get("epochs", 1))
    batch_size = int(model_config.get("batch_size", 1))
    base_filters = int(model_config.get("base_filters", 16))

    class_schema = training_manifest["class_schema"]
    num_classes = max(int(entry["class_index"]) for entry in class_schema) + 1

    image_tensors = []
    label_tensors = []
    for item in training_manifest["items"]:
        image_array = _load_rgb_image(Path(item["source_path"]), (target_width, target_height))
        label_map = np.load(item["label_map_path"], allow_pickle=False).astype(np.uint16)
        resized_label = _resize_label_map(label_map, (target_width, target_height))
        image_tensors.append(image_array)
        label_tensors.append(resized_label)

    x_train = np.stack(image_tensors).astype(np.float32) / 255.0
    y_train = np.stack(label_tensors).astype(np.uint16)
    model = build_unet_model(
        input_shape=(target_height, target_width, 3),
        num_classes=num_classes,
        base_filters=base_filters,
    )
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    model_path = output_dir / "model.keras"
    history_path = output_dir / "history.json"
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    model.save(str(model_path))
    history_path.write_text(json.dumps(history.history, indent=2, sort_keys=True), encoding="utf-8")
    artifact_manifest = {
        "runtime": "tensorflow.keras",
        "family_id": "keras.unet.semantic_segmentation",
        "model_path": str(model_path),
        "history_path": str(history_path),
        "input_size": [target_height, target_width],
        "num_classes": num_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "base_filters": base_filters,
    }
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "runtime": "tensorflow.keras",
        "artifact_manifest_path": str(artifact_manifest_path),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "input_size": [target_height, target_width],
        "num_classes": num_classes,
    }


def predict_unet_from_manifest(prediction_manifest: dict, artifact_manifest_path: Path) -> list[dict]:
    tf = _import_tensorflow()
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(artifact_manifest["model_path"])
    target_height, target_width = artifact_manifest["input_size"]

    prediction_items = []
    for item in prediction_manifest["items"]:
        image = _load_rgb_image(Path(item["source_path"]), (target_width, target_height)).astype(np.float32) / 255.0
        image_batch = np.expand_dims(image, axis=0)
        probabilities = model.predict(image_batch, verbose=0)[0]
        resized_logits = np.argmax(probabilities, axis=-1).astype(np.uint16)
        original_shape = (int(item["height"]), int(item["width"]))
        restored_prediction = _resize_label_map(resized_logits, (original_shape[1], original_shape[0]))
        prediction_items.append(
            {
                "image_id": int(item["image_id"]),
                "annotation_revision_id": int(item["annotation_revision_id"]),
                "raw_prediction": restored_prediction,
                "refined_prediction": restored_prediction.copy(),
            }
        )
    return prediction_items


def _load_rgb_image(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        resized = rgb_image.resize(target_size, Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def _resize_label_map(label_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    label_image = Image.fromarray(np.asarray(label_map, dtype=np.uint16), mode="I;16")
    resized = label_image.resize(target_size, Image.NEAREST)
    return np.asarray(resized, dtype=np.uint16)

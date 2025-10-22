import requests
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import numpy as np
import socket
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import regularizers
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay


def url_capture(url: str) -> str:
    if url.endswith('/capture'):
        return url
    return url.rstrip('/') + '/capture'

def aufnahme(url, interval, klasse, ordner):
    # URL des ESP32-CAM /capture Endpunkts
    url = url_capture(url)

    # Intervall zwischen den Aufnahmen in Sekunden
    interval = interval

    # Pfad zum Speicherordner
    save_path = os.path.join(ordner, klasse)
    os.makedirs(save_path, exist_ok=True)

    while True:
        try:
            # Bild vom /capture Endpunkt abrufen
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Aktuellen Zeitstempel für den Dateinamen
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(save_path, f'capture_{timestamp}.jpg')

                # Bild speichern
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f'Bild gespeichert: {filename}')
            else:
                print(f'Fehler: Statuscode {response.status_code}')
        except requests.RequestException as e:
            print(f'Anfrage fehlgeschlagen: {e}')

        # Warten bis zum nächsten Abruf
        time.sleep(interval)

def get_class_names(train_dir):
    """
    Gibt eine alphabetisch sortierte Liste der Klassenbezeichnungen zurück,
    basierend auf den Unterordnern im angegebenen Trainingsverzeichnis.
    """
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

def training_classification_demo(ordner, model_name, epochen):
    # Datenverzeichnis anpassen
    train_dir = ordner + '/'

    # Bildparameter
    img_height = 240  # Höhe des Bildes
    img_width = 320  # Breite des Bildes
    batch_size = 32

    # Trainings- und Validierungsdatensätze erstellen
    dataset = image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'  # Mehrklassen-Klassifikation
    )
    num_classes = len(dataset.class_names)

    # Dataset normalisieren
    normalization_layer = layers.Rescaling(1. / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    # CNN-Modell definieren
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(240, 320, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flach- und Vollverbundene Schichten
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  # 4 statt 3 Klassen

    # Modellkompilierung
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Modell trainieren
    history = model.fit(dataset, epochs=epochen)  # Weniger Epochen zum Verhindern von Overfitting

    # Modell speichern
    model.save(model_name)

def FFN(ordner, model_name, epochen, n_full, droprate=0.0, resize=None, padding=False, val_ordner=None, aug_parameter=None, alpha=0, lr=0.001, decay=True):
    train_dir = ordner + '/'
    val_dir = val_ordner
    val_set = None

    # irgendeine Datei aus dem Ordner nehmen
    sample_path = os.path.join(train_dir, os.listdir(train_dir)[0],
                               os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size  # PIL gibt (Breite, Höhe)
    print(f"Ermittelte Bildgröße der Trainingsdaten: {img_height}x{img_width}")

    if val_ordner is not None:
        sample_path = os.path.join(val_dir, os.listdir(val_dir)[0],
                                   os.listdir(os.path.join(val_dir, os.listdir(val_dir)[0]))[0])
        with Image.open(sample_path) as img:
            img_width2, img_height2 = img.size  # PIL gibt (Breite, Höhe)
        print(f"Ermittelte Bildgröße der Validierungsdaten: {img_height2}x{img_width2}")


    # Bildparameter
    img_height = img_height
    img_width = img_width
    batch_size = 32

    if resize==None:
        # Trainings- und Validierungsdatensätze erstellen
        dataset = image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'  # Mehrklassen-Klassifikation
        )
        num_classes = len(dataset.class_names)
        print("Datensatz in Originalgröße verarbeitet")
        if val_ordner is not None:
            if padding == False:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden ohne Padding von {img_height2}x{img_width2} auf {img_height}x{img_width} skaliert")
            elif padding == True:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height2, img_width2),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )

                def resize_with_padding(image, label):
                    image = tf.image.resize_with_pad(image, img_height, img_width)
                    return image, label

                val_set = val_set.map(resize_with_padding)
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden mit Padding von {img_height2}x{img_width2} auf {img_height}x{img_width} skaliert")

    elif resize[0] is not None:
        if padding==False:
            dataset = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(resize[0], resize[1]),  # hartes Resize
                batch_size=batch_size,
                label_mode='categorical'
            )
            img_height=resize[0]
            img_width=resize[1]
            num_classes = len(dataset.class_names)
            print("Datensatz resized ohne Padding ", resize)
            if val_ordner is not None:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(resize[0], resize[1]),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden ohne Padding von {img_height2}x{img_width2} auf {resize[0]}x{resize[1]} skaliert")

        elif padding==True:
            dataset = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(img_height, img_width),  # Originalgröße
                batch_size=batch_size,
                label_mode='categorical'
            )
            img_height=resize[0]
            img_width=resize[1]
            print("Datensatz resized mit padding ", resize)

            def resize_with_padding(image, label):
                image = tf.image.resize_with_pad(image, resize[0], resize[1])
                return image, label

            num_classes = len(dataset.class_names)
            dataset = dataset.map(resize_with_padding)
            if val_ordner is not None:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height2, img_width2),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )

                def resize_with_padding(image, label):
                    image = tf.image.resize_with_pad(image, resize[0], resize[1])
                    return image, label

                val_set = val_set.map(resize_with_padding)
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden mit Padding von {img_height2}x{img_width2} auf {resize[0]}x{resize[1]} skaliert")

    else:
        raise ValueError("Wert von resize muss die Form [Höhe,Breite] haben")


    # Dataset normalisieren
    AUTOTUNE = tf.data.AUTOTUNE

    # Normalisieren
    normalization_layer = layers.Rescaling(1. / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                          num_parallel_calls=AUTOTUNE)
    if val_ordner is not None:
        val_set = val_set.map(lambda x, y: (normalization_layer(x), y),
                              num_parallel_calls=AUTOTUNE)

    if aug_parameter is not None:
        # Augmentation nur fürs Training
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip(aug_parameter[0]),
                layers.RandomRotation(aug_parameter[1]),
                layers.RandomZoom(aug_parameter[2]),
                layers.RandomContrast(aug_parameter[3]),
            ],
            name="data_augmentation",
        )

        print("Augmentation aktiv")
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=AUTOTUNE)

    # Pipeline tunen
    dataset = dataset.prefetch(AUTOTUNE)

    initial_lr = lr  # z. B. 1e-3
    card = tf.data.experimental.cardinality(dataset).numpy()  # -1=INFINITE, -2=UNKNOWN
    if card <= 0:
        # Fallback: wenn dir train_samples und batch_size bekannt sind, nimm:
        # steps_per_epoch = math.ceil(train_samples / batch_size)
        raise ValueError("steps_per_epoch konnte nicht bestimmt werden. Bitte angeben oder Fallback setzen.")
    steps_per_epoch = int(card)
    decay_steps = epochen * steps_per_epoch  # Gesamtanzahl Schritte
    dec = 1e-5 / initial_lr
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=dec
    )

    if val_ordner is not None:
        val_set = val_set.prefetch(AUTOTUNE)

    # CNN-Modell definieren
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(img_height, img_width, 3)))

    for i in range(len(n_full)):
        model.add(layers.Dense(n_full[i], kernel_regularizer=regularizers.l2(alpha), use_bias=False, kernel_initializer='he_normal'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(droprate))

    model.add(layers.Dense(num_classes, activation='softmax'))

    # Modellkompilierung
    if decay is True:
        model.compile(optimizer=Adam(learning_rate=lr_schedule),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Mit lr-decay")
    else:
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    if val_ordner is not None:
        history = model.fit(dataset, validation_data=val_set, epochs=epochen)
    else:
        history = model.fit(dataset, epochs=epochen)

    if val_set is not None:
        results = model.evaluate(val_set, verbose=0)
        print(results)
    # Modell speichern
    model.save(model_name)
    print(model.summary())

def training_classification(ordner, model_name, epochen, n_full, pool_size, conv_filter, filter_size, stride=None, droprate=0.0, resize=None, padding=False, val_ordner=None, aug_parameter=None, alpha=0, lr=0.001, decay=True):
    train_dir = ordner + '/'
    val_dir = val_ordner
    val_set = None

    # irgendeine Datei aus dem Ordner nehmen
    sample_path = os.path.join(train_dir, os.listdir(train_dir)[0],
                               os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size  # PIL gibt (Breite, Höhe)
    print(f"Ermittelte Bildgröße der Trainingsdaten: {img_height}x{img_width}")

    if val_ordner is not None:
        sample_path = os.path.join(val_dir, os.listdir(val_dir)[0],
                                   os.listdir(os.path.join(val_dir, os.listdir(val_dir)[0]))[0])
        with Image.open(sample_path) as img:
            img_width2, img_height2 = img.size  # PIL gibt (Breite, Höhe)
        print(f"Ermittelte Bildgröße der Validierungsdaten: {img_height2}x{img_width2}")


    # Bildparameter
    img_height = img_height
    img_width = img_width
    batch_size = 32

    if resize==None:
        # Trainings- und Validierungsdatensätze erstellen
        dataset = image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'  # Mehrklassen-Klassifikation
        )
        num_classes = len(dataset.class_names)
        print("Datensatz in Originalgröße verarbeitet")
        if val_ordner is not None:
            if padding == False:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden ohne Padding von {img_height2}x{img_width2} auf {img_height}x{img_width} skaliert")
            elif padding == True:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height2, img_width2),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )

                def resize_with_padding(image, label):
                    image = tf.image.resize_with_pad(image, img_height, img_width)
                    return image, label

                val_set = val_set.map(resize_with_padding)
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden mit Padding von {img_height2}x{img_width2} auf {img_height}x{img_width} skaliert")

    elif resize[0] is not None:
        if padding==False:
            dataset = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(resize[0], resize[1]),  # hartes Resize
                batch_size=batch_size,
                label_mode='categorical'
            )
            img_height=resize[0]
            img_width=resize[1]
            num_classes = len(dataset.class_names)
            print("Datensatz resized ohne Padding ", resize)
            if val_ordner is not None:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(resize[0], resize[1]),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden ohne Padding von {img_height2}x{img_width2} auf {resize[0]}x{resize[1]} skaliert")

        elif padding==True:
            dataset = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(img_height, img_width),  # Originalgröße
                batch_size=batch_size,
                label_mode='categorical'
            )
            img_height=resize[0]
            img_width=resize[1]
            print("Datensatz resized mit padding ", resize)

            def resize_with_padding(image, label):
                image = tf.image.resize_with_pad(image, resize[0], resize[1])
                return image, label

            num_classes = len(dataset.class_names)
            dataset = dataset.map(resize_with_padding)
            if val_ordner is not None:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height2, img_width2),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )

                def resize_with_padding(image, label):
                    image = tf.image.resize_with_pad(image, resize[0], resize[1])
                    return image, label

                val_set = val_set.map(resize_with_padding)
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden mit Padding von {img_height2}x{img_width2} auf {resize[0]}x{resize[1]} skaliert")

    else:
        raise ValueError("Wert von resize muss die Form [Höhe,Breite] haben")


    # Dataset normalisieren
    AUTOTUNE = tf.data.AUTOTUNE

    # Normalisieren
    normalization_layer = layers.Rescaling(1. / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                          num_parallel_calls=AUTOTUNE)
    if val_ordner is not None:
        val_set = val_set.map(lambda x, y: (normalization_layer(x), y),
                              num_parallel_calls=AUTOTUNE)

    if aug_parameter is not None:
        # Augmentation nur fürs Training
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip(aug_parameter[0]),
                layers.RandomRotation(aug_parameter[1]),
                layers.RandomZoom(aug_parameter[2]),
                layers.RandomContrast(aug_parameter[3]),
            ],
            name="data_augmentation",
        )

        print("Augmentation aktiv")
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=AUTOTUNE)

    # Pipeline tunen
    dataset = dataset.prefetch(AUTOTUNE)

    initial_lr = lr  # z. B. 1e-3
    card = tf.data.experimental.cardinality(dataset).numpy()  # -1=INFINITE, -2=UNKNOWN
    if card <= 0:
        # Fallback: wenn dir train_samples und batch_size bekannt sind, nimm:
        # steps_per_epoch = math.ceil(train_samples / batch_size)
        raise ValueError("steps_per_epoch konnte nicht bestimmt werden. Bitte angeben oder Fallback setzen.")
    steps_per_epoch = int(card)
    decay_steps = epochen * steps_per_epoch  # Gesamtanzahl Schritte
    dec = 1e-5 / initial_lr
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=dec
    )

    if val_ordner is not None:
        val_set = val_set.prefetch(AUTOTUNE)

    # CNN-Modell definieren
    model = models.Sequential()
    model.add(layers.Input(shape=(img_height, img_width, 3)))

    """
    if aug_parameter is not None:
        model.add(layers.GaussianNoise(aug_parameter[4]))
    """
    #model.add(layers.MaxPooling2D((pool_size, pool_size)))
    for i in range(len(conv_filter)):
        model.add(layers.Conv2D(
            conv_filter[i],
            (filter_size, filter_size),
            strides=(1, 1),
            #padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(alpha),
            kernel_initializer='he_normal'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((pool_size, pool_size)))

    #model.add(layers.Flatten())
    model.add(layers.GlobalAveragePooling2D(name="gap"))
    for i in range(len(n_full)):
        model.add(layers.Dense(n_full[i], use_bias=False, kernel_initializer='he_normal'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(droprate))

    model.add(layers.Dense(num_classes, activation='softmax'))  # 4 statt 3 Klassen

    # Modellkompilierung
    if decay is True:
        model.compile(optimizer=Adam(learning_rate=lr_schedule),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Mit lr-decay")
    else:
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    if val_ordner is not None:
        history = model.fit(dataset, validation_data=val_set, epochs=epochen)
    else:
        history = model.fit(dataset, epochs=epochen)

    if val_set is not None:
        results = model.evaluate(val_set, verbose=0)
        print(results)
    # Modell speichern
    model.save(model_name)
    print(model.summary())

def validation_classification(model, val_ordner, padding=False):
    # --- Bild-Params (wie im Training) ---
    sample_path = os.path.join(val_ordner, os.listdir(val_ordner)[0],
                               os.listdir(os.path.join(val_ordner, os.listdir(val_ordner)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size  # PIL gibt (Breite, Höhe)
    print(f"Ermittelte Bildgröße der Validierungsbilder: {img_height}x{img_width}")

    # --- Modell laden ---
    model = keras.models.load_model(model)
    print(f"Ermittelte Bildgröße des Modells: {model.input_shape[1]}x{model.input_shape[2]}")
    img_size = (model.input_shape[1], model.input_shape[2])
    batch_size = 32

    if padding==False:
        val_ds = keras.utils.image_dataset_from_directory(
            val_ordner,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            label_mode="int"
        )
        class_names = val_ds.class_names
        print(f"Größe Bilder im Datensatz: {img_height}x{img_width} vs. im Modell: {img_size}")
    elif padding==True:
        val_ds = keras.utils.image_dataset_from_directory(
            val_ordner,
            image_size=(img_height,img_width),
            batch_size=batch_size,
            shuffle=False,
            label_mode="int"
        )
        class_names = val_ds.class_names
        print(f"Datensatz von {img_height}x{img_width} resized mit padding auf {img_size}")

        def resize_with_padding(image, label):
            image = tf.image.resize_with_pad(image, img_size[0], img_size[1])
            return image, label

        val_ds = val_ds.map(resize_with_padding)

    print("Klassen (aus val_daten abgeleitet):", class_names)

    # --- gleiche Vorverarbeitung wie im Training: Resize + Rescaling(1/255) ---
    normalization = keras.layers.Rescaling(1. / 255)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # --- Vorhersagen einsammeln ---
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Metriken ---
    acc = accuracy_score(y_true, y_pred)
    f1_weight = f1_score(y_true, y_pred, average="weighted")
    f1_per_cls = f1_score(y_true, y_pred, average=None)  # array in Klassenreihenfolge

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"F1 (weighted):       {f1_weight:.4f}")
    print("\nF1 pro Klasse:")
    for name, f1c in zip(class_names, f1_per_cls):
        print(f"  {name:>12}: {f1c:.4f}")
    return acc

def testen_classification(url, model, ordner, padding=False, live=False, interval=3):
    url = url_capture(url)
    model = keras.models.load_model(model)
    # Bildparameter
    img_height = model.input_shape[1]
    img_width = model.input_shape[2]

    # Klassenbezeichnungen
    class_names = get_class_names(ordner)
    print(class_names)

    def capture_image():
        """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
        filename = 'latest_pic.jpg'
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f'Bild gespeichert: {filename}')
                with Image.open(filename) as img:
                    width, height = img.size  # PIL gibt (Breite, Höhe)
                print(f"Ermittelte Bildgröße der Validierungsbilder: {height}x{width}")
                return filename, height, width
            else:
                print(f'Fehler: Statuscode {response.status_code}')
        except requests.RequestException as e:
            print(f'Anfrage fehlgeschlagen: {e}')
        return None

    def classify_image(image_path, height, width):

        """Klassifiziert ein einzelnes Bild mit dem geladenen Modell."""
        if padding==False:
            img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
            img.save("latest_pic_not_padded.jpg")
            if(img_height==height and img_width==width):
                print(f"Bilder besitzen die richtige Größe {img_height}x{img_width}")
            else:
                print(f"Bilder werden ohne Padding von {height}x{width} auf {img_height}x{img_width} skaliert")

        elif padding==True:
            img = keras.preprocessing.image.load_img(image_path)
            img = keras.preprocessing.image.img_to_array(img)
            img = tf.image.resize_with_pad(img, img_height, img_width)
            img_to_save = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()
            Image.fromarray(img_to_save).save("latest_pic_padded.jpg")

            if(img_height==height and img_width==width):
                print(f"Bilder besitzen die richtige Größe {img_height}x{img_width}")
            else:
                print(f"Bilder werden mit Padding von {height}x{width} auf {img_height}x{img_width} skaliert")

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Index der höchsten Wahrscheinlichkeit
        confidence = np.max(predictions[0])  # Höchste Wahrscheinlichkeit
        print(f"Rohwerte der Vorhersage: {predictions[0]}")
        return class_names[predicted_class], confidence

    # Bild aufnehmen und klassifizieren
    while True:
        image_path, height, width = capture_image()
        if image_path:
            result, confidence = classify_image(image_path, height, width)
            print(f"Das Bild enthält: {result} (Sicherheit: {confidence:.2f})")
            if live==False:
                return result, confidence
        if live==False:
            return None
        time.sleep(interval)

def neural_network_classification(url, arduino_ip, ordner, port, model, padding=False):
    arduino_ip = arduino_ip  # IP-Adresse des Arduino
    port = port  # Muss mit der Portnummer im Arduino-Sketch übereinstimmen
    url = url_capture(url)

    # Socket erstellen
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((arduino_ip, port))

    received_values = []  # Liste zum Speichern der empfangenen Werte

    ################################################################################

    # ESP32-CAM Endpunkt
    esp32_url = url

    # Bildparameter
    sample_path = os.path.join(ordner, os.listdir(ordner)[0],
                               os.listdir(os.path.join(ordner, os.listdir(ordner)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size  # PIL gibt (Breite, Höhe)
    print(f"Ermittelte Bildgröße der Validierungsbilder: {img_height}x{img_width}")

    # Klassenbezeichnungen
    class_names = get_class_names(ordner)
    print(class_names)

    # Modell laden
    model = keras.models.load_model(model)

    def capture_image():
        """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
        filename = 'latest_pic.jpg'
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f'Bild gespeichert: {filename}')
                with Image.open(filename) as img:
                    width, height = img.size  # PIL gibt (Breite, Höhe)
                print(f"Ermittelte Bildgröße der Aufnahme: {height}x{width}")
                return filename, height, width
            else:
                print(f'Fehler: Statuscode {response.status_code}')
        except requests.RequestException as e:
            print(f'Anfrage fehlgeschlagen: {e}')
        return None

    def classify_image(image_path, height, width):

        """Klassifiziert ein einzelnes Bild mit dem geladenen Modell."""
        if padding==False:
            img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
            img.save("latest_pic_not_padded.jpg")
            if(img_height==height and img_width==width):
                print(f"Bilder besitzen die richtige Größe {img_height}x{img_width}")
            else:
                print(f"Bilder werden ohne Padding von {height}x{width} auf {img_height}x{img_width} skaliert")

        elif padding==True:
            img = keras.preprocessing.image.load_img(image_path)
            img = keras.preprocessing.image.img_to_array(img)
            img = tf.image.resize_with_pad(img, img_height, img_width)
            img_to_save = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()
            Image.fromarray(img_to_save).save("latest_pic_padded.jpg")

            if(img_height==height and img_width==width):
                print(f"Bilder besitzen die richtige Größe {img_height}x{img_width}")
            else:
                print(f"Bilder werden mit Padding von {height}x{width} auf {img_height}x{img_width} skaliert")

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Index der höchsten Wahrscheinlichkeit
        confidence = np.max(predictions[0])  # Höchste Wahrscheinlichkeit
        print(f"Rohwerte der Vorhersage: {predictions[0]}")
        return class_names[predicted_class], confidence

    # Bild aufnehmen und klassifizieren
    image_path, height, width = capture_image()
    if image_path:
        result, confidence = classify_image(image_path, height, width)
        print(f"Das Bild enthält: {result} (Sicherheit: {confidence:.2f})")

    #########################################################################################################

    try:
        while True:
            # Daten vom Arduino empfangen
            response = client_socket.recv(1024).decode().strip()

            if response:  # Falls eine Antwort empfangen wurde
                try:
                    value = int(response)  # Versuche, die Antwort in eine Zahl umzuwandeln
                    received_values.append(value)  # Wert in die Liste speichern
                    print(f"Empfangen und gespeichert: {value}")
                    if (value == 42):
                        image_path, height, width = capture_image()
                        if image_path:
                            result, confidence = classify_image(image_path, height, width)
                            result_str = f"{result},{confidence}\n"
                            print(f"Das Bild enthält: {result} (Sicherheit: {confidence:.2f})")
                            client_socket.sendall(result_str.encode())

                except ValueError:
                    print(f"Fehler beim Konvertieren: {response}")  # Falls ungültige Daten empfangen werden

            # Falls du eine Bedingung zum Stoppen willst (z. B. nach 10 Werten)
            if len(received_values) >= 10:
                print("Genug Werte empfangen, beende die Verbindung.")
                break
    finally:
        client_socket.close()

def training_detection(dataset, epochen, img_size=(640,640)):

    # YOLO-Modell laden
    model = YOLO('yolov8n.pt')

    # Training starten
    results = model.train(data=os.path.join(dataset.location, 'data.yaml'), epochs=epochen, imgsz=img_size)

    # Modell auf TEST-Daten evaluieren
    metrics = model.val(data=os.path.join(dataset.location, 'data.yaml'), split='test')

    # Vorhersagen auf Testdaten visualisieren
    test_images_path = os.path.join(dataset.location, 'test', 'images')
    model.predict(source=test_images_path, conf=0.2, save=True)

def capture_image(url):
    """Nimmt ein einzelnes Bild von der ESP32-CAM auf und speichert es unter dem gleichen Namen."""
    filename = 'latest_capture.jpg'
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f'Bild gespeichert: {filename}')
            return filename
        else:
            print(f'Fehler: Statuscode {response.status_code}')
    except requests.RequestException as e:
        print(f'Anfrage fehlgeschlagen: {e}')
    return None

def testen_detection(url, model, conf_thresh, img_size=None):
    url = url_capture(url)
    model = YOLO(model)
    capture_image(url)
    if img_size is None:
        results = model.predict(source='latest_capture.jpg', conf=conf_thresh)
    else:
        results = model.predict(source='latest_capture.jpg', imgsz=img_size, conf=conf_thresh)

    # Ergebnisse extrahieren (Liste von Result-Objekten, hier nur 1 Bild → results[0])
    result = results[0]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])  # Klassenindex (z. B. 0 für „ball“)
        label = model.names[cls_id]  # Klassennamen (z. B. "ball")
        conf = float(box.conf[0])  # Konfidenzscore
        xyxy = box.xyxy[0].tolist()  # Bounding Box Koordinaten [x1, y1, x2, y2]
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": xyxy
        })
    print(detections)

    annotated_img = result.plot()
    annotated_pil = Image.fromarray(annotated_img)
    annotated_pil.save('latest_capture_annotated.jpg')

    # 3 Sekunden warten
    if detections:
        # Nur das erste erkannte Objekt verwenden
        det = detections[0]
        label = det['label']
        conf = round(det['confidence'], 2)
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
    else:
        # Kein Objekt erkannt
        label = "none"
        conf = 0
        x1 = y1 = x2 = y2 = 0

    # Arduino-freundlicher, einfacher CSV-String
    result_str = f"{label},{conf},{x1},{y1},{x2},{y2}\n"
    return result_str

def neural_network_detection(url, arduino_ip, port, model, conf_thresh):
    arduino_ip = arduino_ip  # IP-Adresse des Arduino
    port = port  # Muss mit der Portnummer im Arduino-Sketch übereinstimmen
    url = url_capture(url)

    # Socket erstellen
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((arduino_ip, port))

    received_values = []  # Liste zum Speichern der empfangenen Werte

    # Pfad zum Speicherordner
    save_path = 'daten2/obj'

    # Überprüfen, ob der Speicherordner existiert, andernfalls erstellen
    os.makedirs(save_path, exist_ok=True)
    model = YOLO(model)

    def classify_image(filename):
        results = model.predict(source=filename, conf=conf_thresh)
        # Ergebnisse extrahieren (Liste von Result-Objekten, hier nur 1 Bild → results[0])
        result = results[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Klassenindex (z. B. 0 für „ball“)
            label = model.names[cls_id]  # Klassennamen (z. B. "ball")
            conf = float(box.conf[0])  # Konfidenzscore
            xyxy = box.xyxy[0].tolist()  # Bounding Box Koordinaten [x1, y1, x2, y2]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": xyxy
            })
        print(detections)

        annotated_img = result.plot()
        annotated_pil = Image.fromarray(annotated_img)
        annotated_pil.save('annotated_latest.jpg')

        # 3 Sekunden warten
        if detections:
            compact = []
            for d in detections:
                compact.append({
                    "label": d["label"],
                    "confidence": round(float(d["confidence"]), 2),  # runden um Platz zu sparen
                    "bbox": [int(coord) for coord in d["bbox"]]  # optional: ints statt floats
                })

            json_str = json.dumps(compact, separators=(',', ':'))  # compact JSON
            # Einfach eine Zeilen-terminierte Nachricht senden
            client_socket.sendall((json_str + "\n").encode())
        else:
            # Kein Objekt erkannt
            compact = []
            compact.append({
                "label": "none",
                "confidence": 0,  # runden um Platz zu sparen
                "bbox": [0.0,0.0,0.0,0.0] # optional: ints statt floats
            })
            json_str = json.dumps(compact, separators=(',', ':'))  # compact JSON
            # Einfach eine Zeilen-terminierte Nachricht senden
            client_socket.sendall((json_str + "\n").encode())

    #########################################################################################################

    try:
        while True:
            # Daten vom Arduino empfangen
            response = client_socket.recv(1024).decode().strip()

            if response:  # Falls eine Antwort empfangen wurde
                try:
                    value = int(response)  # Versuche, die Antwort in eine Zahl umzuwandeln
                    received_values.append(value)  # Wert in die Liste speichern
                    print(f"Empfangen und gespeichert: {value}")
                    if (value == 42):
                        image_path = capture_image(url)
                        if image_path:
                            classify_image(image_path)

                except ValueError:
                    print(f"Fehler beim Konvertieren: {response}")  # Falls ungültige Daten empfangen werden

            # Falls du eine Bedingung zum Stoppen willst (z. B. nach 10 Werten)
            if len(received_values) >= 300:
                print("Genug Werte empfangen, beende die Verbindung.")
                break

    finally:
        client_socket.close()



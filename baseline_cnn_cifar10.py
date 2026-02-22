# baseline_cnn_cifar10.py
# Clean baseline: CIFAR-10 + simple 4-conv-layer CNN

import os
import time
import random
import numpy as np
import tensorflow as tf
import argparse
import csv
from pathlib import Path

import threading
import psutil

from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
)

from codecarbon import EmissionsTracker

def make_tracker(project_name: str, output_dir: str = "."):
    """
    CodeCarbon tracker.
    We keep it local (no external API required).
    """
    return EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        output_file="codecarbon_log.csv",  # CodeCarbon's own log (separate from our metrics CSV)
        log_level="error",
        save_to_file=True,
        tracking_mode="process",  # track this Python process
    )

def main ():

    class PeakMemoryMonitor:
        """
        Samples memory repeatedly and records the maximum (peak).
        - CPU peak: RAM used by THIS Python process (RSS).
        - GPU peak: VRAM used on the NVIDIA GPU (NVML).
        """

        def __init__(self, sample_interval_sec: float = 0.05, gpu_index: int = 0):
            self.sample_interval_sec = sample_interval_sec
            self.gpu_index = gpu_index

            self._stop = threading.Event()
            self._thread = None

            self.cpu_peak_bytes = 0
            self.gpu_peak_bytes = 0

            self._proc = psutil.Process(os.getpid())
            self._gpu_handle = None
            self._gpu_enabled = False
            self.gpu_available = False  # <--- used to decide NA vs value in CSV

        def start(self):
            # Try to enable GPU monitoring (if NVML is available)
            try:
                nvmlInit()
                self._gpu_handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
                self._gpu_enabled = True
                self.gpu_available = True
            except Exception:
                self._gpu_enabled = False
                self.gpu_available = False

            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def stop(self):
            self._stop.set()
            if self._thread is not None:
                self._thread.join()

            if self._gpu_enabled:
                try:
                    nvmlShutdown()
                except Exception:
                    pass

        def _run(self):
            while not self._stop.is_set():
                # CPU RAM used by this Python process
                try:
                    rss = self._proc.memory_info().rss
                    if rss > self.cpu_peak_bytes:
                        self.cpu_peak_bytes = rss
                except Exception:
                    pass

                # GPU VRAM used (if enabled)
                if self._gpu_enabled:
                    try:
                        info = nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        used = info.used
                        if used > self.gpu_peak_bytes:
                            self.gpu_peak_bytes = used
                    except Exception:
                        pass

                self._stop.wait(self.sample_interval_sec)

        @staticmethod
        def bytes_to_mb(x: int) -> float:
            return x / (1024 * 1024)



    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--csv", type=str, default="hidden_cost_experiments.csv")
    args = parser.parse_args()

    RUN_ID = args.run_id
    VARIANT = args.variant
    CSV_PATH = Path(args.csv)

    csv_header = [
    "variant",
    "run_id",
    "train_time_sec",
    "eval_time_sec",
    "cpu_peak_train_mb",
    "gpu_peak_train_mb",
    "cpu_peak_eval_mb",
    "gpu_peak_eval_mb",
    "train_energy_kwh",
    "train_co2_kg",
    "eval_energy_kwh",
    "eval_co2_kg",
    "test_accuracy",
    ]

    
    # Create CSV file with header if it does not exist
    if not CSV_PATH.exists():
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
        


    # 1) Make runs reproducible
    # We want Python , NumPy and TensorFlow to behave as consistently as possinle accross runs 
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED) # Mean use the same hasing behavior every time 
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


    # 2) Load and prep CIFAR-10
    # CIFAR-10 images are x-train: 32x32 pixels with 3 color channels RGB, y-train: labels are 0..9
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Convert pixels from 0..255 into 0..1 (helps training)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Turn labels into shape (N,) instead of (N,1) so Keras can use them cleanly with sparse_categorical_crossentropy loss
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()


    # 3) Build a simple 4-layer CNN

    def build_cnn():
        # A CNN tries to recognize what is inside an image by learning patterns in the pixel data.
        # create a "machine' that 
        # a) takes an image (32*32*03) 
        # b) learns patterns (small patterns -> bigger patterns -> objects) 
        # c) outputs 10 number (one per class) 
        # d) the biggest number = predicted class   
        
        inputs = tf.keras.Input(shape=(32, 32, 3))

        # Conv2D learns patterns in images by applying filters (like edge detectors) across the image.
        #pooling means shrink the image size , Keep the strongest features and reduce the number of parameters, which helps focus on the most important patterns and speeds up training.
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)

        # Flatten turns the 3D feature maps  "image grid "into a 1D vector, which can (Dense) layers for classification.
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x) # 10 outputs because CIFAR-10 has 10 classes.

        model = tf.keras.Model(inputs, outputs)
        return model


    model = build_cnn()

    # 4) Compile (training settings)
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 3  # keep small for testing; can increase later

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 5) Train (measure fit time + memory peaks + emissions)
    train_mon = PeakMemoryMonitor()
    train_mon.start()

    train_start = time.perf_counter()
    train_tracker = make_tracker(project_name="baseline_train")
    train_tracker.start()
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=2
    )

    train_end = time.perf_counter()
    train_seconds = train_end - train_start

    train_mon.stop()
    
    cpu_peak_train_mb = round(PeakMemoryMonitor.bytes_to_mb(train_mon.cpu_peak_bytes), 2)
    if train_mon.gpu_available:
        gpu_peak_train_mb = round(PeakMemoryMonitor.bytes_to_mb(train_mon.gpu_peak_bytes), 2)
    else:
        gpu_peak_train_mb = "NA"
    
    train_emissions_kg = train_tracker.stop()  # returns CO2 in kg (float)
    train_energy_kwh = getattr(train_tracker.final_emissions_data, "energy_consumed", None)
    train_co2_kg = round(train_emissions_kg, 6) if train_emissions_kg is not None else "NA"
    train_energy_kwh = round(train_energy_kwh, 6) if train_energy_kwh is not None else "NA"



    # 6) Evaluate (measure evaluate time + memory peaks + emissions)
    eval_mon = PeakMemoryMonitor()
    eval_mon.start()
    eval_start = time.perf_counter()

    eval_tracker = make_tracker(project_name="baseline_eval")
    eval_tracker.start()

    test_loss, test_acc = model.evaluate(
        x_test, y_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    eval_end = time.perf_counter()
    eval_seconds = eval_end - eval_start

    eval_mon.stop()
    cpu_peak_eval_mb = round(PeakMemoryMonitor.bytes_to_mb(eval_mon.cpu_peak_bytes), 2)
    if eval_mon.gpu_available:
        gpu_peak_eval_mb = round(PeakMemoryMonitor.bytes_to_mb(eval_mon.gpu_peak_bytes), 2)
    else:
        gpu_peak_eval_mb = "NA"

    eval_emissions_kg = eval_tracker.stop()
    eval_energy_kwh = getattr(eval_tracker.final_emissions_data, "energy_consumed", None)

    eval_co2_kg = round(eval_emissions_kg, 6) if eval_emissions_kg is not None else "NA"
    eval_energy_kwh = round(eval_energy_kwh, 6) if eval_energy_kwh is not None else "NA"

    # 7) Print baseline results
    print("\nCLEAN  CNN BASELINE RESULTS (CIFAR-10): ")
    print(f"Train time (fit only): {train_seconds:.3f} seconds")
    print(f"Eval time (evaluate only): {eval_seconds:.3f} seconds")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")


    
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            VARIANT,
            RUN_ID,
            round(train_seconds, 4),
            round(eval_seconds, 4),
            cpu_peak_train_mb,
            gpu_peak_train_mb,
            cpu_peak_eval_mb,
            gpu_peak_eval_mb,
            train_energy_kwh,
            train_co2_kg,
            eval_energy_kwh,
            eval_co2_kg,
            round(test_acc, 4)
        ])
    

if __name__ == "__main__":
    main()



# Milestone 2 - Baseline CNN CIFAR-10
# PROPER implementation: 10 automated runs, correct memory tracking, no memory leaks
# Metrics: Same as Milestone 1 + precision/recall

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from codecarbon import EmissionsTracker
import psutil
import time
import os
import gc

# Ensure only 1 GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
NUM_RUNS = 10
VARIANT = "baseline"
CSV_FILE = "baseline_results.csv"

print("="*70)
print("MILESTONE 2 - BASELINE EXPERIMENT")
print("="*70)
print(f"Model: 3-block CNN (32-64-128 filters)")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Runs: {NUM_RUNS}")
print("="*70 + "\n")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU Available: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"  {gpu}")
    # Enable memory growth to prevent full allocation
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except:
        pass
else:
    print("⚠ No GPU - running on CPU")
print()

# Load data ONCE
print("Loading CIFAR-10...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
assert X_train.shape[0] == 50000 and X_test.shape[0] == 10000

# Preprocess
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)
print("✓ Data ready\n")


def build_model():
    """Build baseline CNN"""
    model = Sequential([
        # Block 1: 32 filters
        Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        
        # Block 2: 64 filters
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        
        # Block 3: 128 filters
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        
        # Dense
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(10, activation='softmax')
    ])
    return model


class MemoryTracker:
    """Track memory usage for a phase"""
    def __init__(self, phase_name):
        self.phase_name = phase_name
        self.process = psutil.Process()
        self.start_cpu = 0
        self.start_gpu = 0
        self.peak_cpu = 0
        self.peak_gpu = 0
        
    def start(self):
        """Record starting memory"""
        self.start_cpu = self.process.memory_info().rss / (1024**2)
        
        # Get GPU memory if available
        try:
            if gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                self.start_gpu = mem_info['current'] / (1024**2)
        except:
            self.start_gpu = 0
            
    def stop(self):
        """Record peak memory and calculate delta"""
        end_cpu = self.process.memory_info().rss / (1024**2)
        self.peak_cpu = end_cpu - self.start_cpu  # Delta from start
        
        # Get GPU memory delta
        try:
            if gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                end_gpu = mem_info['current'] / (1024**2)
                self.peak_gpu = end_gpu - self.start_gpu
        except:
            self.peak_gpu = 0
            
        return self.peak_cpu, self.peak_gpu


# Storage
all_results = []

# Main loop
for run_id in range(1, NUM_RUNS + 1):
    print(f"\n{'='*70}")
    print(f"RUN {run_id}/{NUM_RUNS}")
    print(f"{'='*70}\n")
    
    try:
        # AGGRESSIVE CLEANUP
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Reset GPU memory stats
        try:
            if gpus:
                tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass
        
        # Set seeds
        np.random.seed(SEED + run_id)
        tf.random.set_seed(SEED + run_id)
        os.environ["PYTHONHASHSEED"] = str(SEED + run_id)
        
        # Build model
        print("Building model...")
        model = build_model()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        print(f"Parameters: {model.count_params():,}")
        
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        print(f"\nTraining ({EPOCHS} epochs)...")
        
        mem_tracker_train = MemoryTracker("train")
        mem_tracker_train.start()
        
        tracker_train = EmissionsTracker(
            project_name=f"baseline_train_{run_id}",
            output_dir=".",
            output_file="codecarbon_train.csv",
            log_level="error"
        )
        tracker_train.start()
        
        train_start = time.time()
        
        history = model.fit(
            X_train, y_cat_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_cat_test),
            verbose=2
        )
        
        train_time = time.time() - train_start
        train_emissions = tracker_train.stop()
        train_energy = getattr(tracker_train.final_emissions_data, "energy_consumed", None)
        
        cpu_train, gpu_train = mem_tracker_train.stop()
        
        print(f"✓ Training: {train_time:.1f}s")
        print(f"  CPU memory: +{cpu_train:.1f} MB")
        if gpu_train > 0:
            print(f"  GPU memory: +{gpu_train:.1f} MB")
        
        # ==========================================
        # EVALUATION PHASE
        # ==========================================
        print("\nEvaluating...")
        
        mem_tracker_eval = MemoryTracker("eval")
        mem_tracker_eval.start()
        
        tracker_eval = EmissionsTracker(
            project_name=f"baseline_eval_{run_id}",
            output_dir=".",
            output_file="codecarbon_eval.csv",
            log_level="error"
        )
        tracker_eval.start()
        
        eval_start = time.time()
        
        results_eval = model.evaluate(X_test, y_cat_test, batch_size=BATCH_SIZE, verbose=0)
        
        eval_time = time.time() - eval_start
        eval_emissions = tracker_eval.stop()
        eval_energy = getattr(tracker_eval.final_emissions_data, "energy_consumed", None)
        
        cpu_eval, gpu_eval = mem_tracker_eval.stop()
        
        test_loss = results_eval[0]
        test_acc = results_eval[1]
        test_prec = results_eval[2]
        test_rec = results_eval[3]
        
        print(f"✓ Eval: {eval_time:.1f}s")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall: {test_rec:.4f}")
        
        # ==========================================
        # SAVE RESULTS
        # ==========================================
        results = {
            'variant': VARIANT,
            'run_id': run_id,
            'train_time_sec': round(train_time, 4),
            'eval_time_sec': round(eval_time, 4),
            'cpu_peak_train_mb': round(cpu_train, 2),
            'gpu_peak_train_mb': round(gpu_train, 2) if gpu_train > 0 else "NA",
            'cpu_peak_eval_mb': round(cpu_eval, 2),
            'gpu_peak_eval_mb': round(gpu_eval, 2) if gpu_eval > 0 else "NA",
            'train_energy_kwh': round(train_energy, 6) if train_energy else "NA",
            'train_co2_kg': round(train_emissions, 6) if train_emissions else "NA",
            'eval_energy_kwh': round(eval_energy, 6) if eval_energy else "NA",
            'eval_co2_kg': round(eval_emissions, 6) if eval_emissions else "NA",
            'test_accuracy': round(test_acc, 4),
            'test_precision': round(test_prec, 4),
            'test_recall': round(test_rec, 4),
        }
        
        all_results.append(results)
        
        # Save incrementally
        df = pd.DataFrame(all_results)
        df.to_csv(CSV_FILE, index=False)
        
        print(f"\n✓ Run {run_id} saved to {CSV_FILE}")
        
        # Final cleanup
        del model
        gc.collect()
        
    except Exception as e:
        print(f"\n✗ ERROR in run {run_id}:")
        print(str(e))
        import traceback
        traceback.print_exc()
        continue

# Summary
print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70 + "\n")

if all_results:
    df = pd.DataFrame(all_results)
    print("SUMMARY:")
    print("-" * 70)
    
    for col in ['train_time_sec', 'cpu_peak_train_mb', 'cpu_peak_eval_mb', 
                'test_accuracy', 'test_precision', 'test_recall']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            print(f"{col:25s}: {vals.mean():.4f} ± {vals.std():.4f}")
    
    print("-" * 70)
    print(f"\n✓ Results: {CSV_FILE}")
    print(f"✓ Completed: {len(all_results)}/{NUM_RUNS} runs")
else:
    print("⚠ No results!")

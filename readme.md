# Hybrid Quantum–Classical Face Verification

This folder contains the full pipeline for our project:
face preprocessing, a classical ResNet-18 baseline, hybrid quantum–classical models, noisy variants, ensembles, and evaluation.

---

## File Overview and Code Structure

### `preprocess.py`

Face detection, alignment, and export script (MTCNN-based).

**Main responsibilities**

- Walk an input directory tree and find all image files.
- Run a face detector on each image and select one or more faces.
- Align and crop faces to a fixed size and save them to an output directory with the same subfolder structure as the input.
- Optionally provide a small helper to build a PyTorch `DataLoader` over the aligned faces.

**Key components**

- **Argument parsing / entry point**
  - Uses `argparse` to read:
    - `--in`: input root directory.
    - `--out`: output root directory.
    - `--size`: output crop size (default 224).
    - `--margin`: extra padding around the detected face bounding box.
    - `--keep-all`: whether to keep all faces per image or only the largest one.
    - `--rebuild`: whether to overwrite existing outputs.

- **Preprocessing pipeline**
  - Iterates over all images under `--in`.
  - For each image:
    - Loads the image.
    - Runs MTCNN to detect face bounding boxes and landmarks.
    - Selects either the largest face or all faces (controlled by `--keep-all`).
    - Aligns the face(s) using the predicted landmarks.
    - Crops and resizes each face to `size × size`.
    - Saves the cropped faces under `--out`, preserving class / identity folders.

- **Dataloader helper (optional)**
  - A helper function that:
    - Takes the aligned-face root directory.
    - Wraps it in a `torchvision.datasets.ImageFolder`.
    - Applies ImageNet-style transforms (to tensor + normalization).
    - Returns a `DataLoader` to be used in the training notebooks.

**Example usage**

```bash
# Standard: one aligned face per image
python preprocess.py --in data/raw_faces --out data/aligned_faces

# Generic paths and keep all detected faces
python preprocess.py --in input_path --out output_path
```

---

### `classical.ipynb`

Trains the purely classical ResNet-18 baseline and exports reusable checkpoints.

**High-level flow**

1. Set up environment and configuration.
2. Build training and validation dataloaders from aligned faces.
3. Define the ResNet-18 model with a task-specific classification head.
4. Implement training / evaluation utilities.
5. Phase A: train the backbone and save a backbone-only checkpoint.
6. Phase B: warm up a new head, then fine-tune the entire model.
7. Optionally, define a small inference helper for single-image prediction.

**Main sections / functions**

- **Imports and global config**
  - Imports PyTorch, Torchvision, and utility modules.
  - Defines device selection.
  - Fixes ImageNet mean and standard deviation for normalization.

- **Dataloaders**
  - `make_dataloaders(train_dir, val_dir, batch_size, num_workers, aug=True)`:
    - Wraps `torchvision.datasets.ImageFolder`.
    - Uses basic augmentations for training.
    - Always applies normalization and converts images to tensors.
  - Alternative “Phase 6” block:
    - Uses a single `DATA_DIR` with or without explicit `train` / `val` subfolders.
    - If needed, performs an 80/20 random split of one dataset into train and validation.
    - Computes class names and, for binary tasks, a `POS_WEIGHT` for imbalanced data.

- **Model definition**
  - `build_resnet18(num_classes)`:
    - Loads an ImageNet-pretrained ResNet-18.
    - Replaces the final `fc` layer:
      - Binary: `Dropout + Linear(in_features, 1)` (for `BCEWithLogitsLoss`).
      - Multi-class: `Dropout + Linear(in_features, num_classes)` (for `CrossEntropyLoss`).
  - `freeze_fc_only(model)`:
    - Sets `requires_grad = False` for all parameters of the final `fc` layer.
  - `save_backbone_only(model, path, img_size=224)`:
    - Removes all `fc.*` keys from the `state_dict`.
    - Saves only the convolutional backbone plus basic metadata (image size, mean, std).

- **Training utilities**
  - `evaluate(model, loader, device, num_classes)`:
    - Runs in evaluation mode.
    - Computes average loss and accuracy on a dataloader.
    - Handles both binary and multi-class cases.
  - `train_one_phase(model, train_loader, val_loader, device, num_classes, ...)`:
    - Generic training loop:
      - Uses `AdamW` optimizer with weight decay.
      - Uses a cosine learning-rate schedule with an adjustable minimum LR.
      - Supports label smoothing for multi-class, and `pos_weight` for binary tasks.
      - Uses mixed precision on CUDA via `torch.cuda.amp` for speed.
      - Implements early stopping based on validation accuracy and restores the best weights.

- **Phase A: backbone-only training**
  - Builds a ResNet-18 model for the task.
  - Freezes the final `fc` layer; only the backbone weights are updated.
  - Trains for several epochs using `train_one_phase`.
  - Saves a backbone-only checkpoint under `outputs/`.

- **Phase B: warm-up and fine-tune**
  - **Head warm-up:**
    - Rebuilds ResNet-18, attaches a new head, and loads backbone weights from Phase A.
    - Freezes the backbone and trains only the new head with a short training phase.
  - **Full fine-tuning:**
    - Unfreezes all parameters.
    - Creates two parameter groups in `AdamW`:
      - One for `fc.*` with a larger learning rate and no weight decay.
      - One for the backbone with a smaller learning rate and regular weight decay.
    - Trains end-to-end with cosine scheduling and early stopping.
    - Saves the final full model checkpoint under `outputs/`.

- **Inference helper**
  - Defines a small function that:
    - Loads an image from disk.
    - Applies the same normalization.
    - Runs the model and returns the predicted class and confidence.

---

### `hybrid.ipynb`

Implements and trains a single hybrid quantum–classical model.

**High-level flow**

1. Load the ResNet-18 backbone checkpoint from `classical.ipynb`.
2. Build a hybrid model where:
   - A frozen ResNet-18 provides high-level features.
   - A small classical projection maps features to a low-dimensional vector.
   - A variational quantum circuit processes this vector.
   - A small classical head maps VQC outputs to class logits.
3. Train the hybrid model and store its checkpoint.

**Main components**

- **Backbone feature extractor**
  - Loads the backbone-only checkpoint.
  - Uses ResNet-18 up to the global average pool layer to yield a 512-dimensional vector.
  - Usually keeps backbone weights frozen to focus training on the quantum head.

- **Quantum circuit definition**
  - Uses a quantum ML framework (PennyLane).
  - Defines a QNode that:
    - Takes a 4-dimensional classical input.
    - Encodes it into the states of a 4-qubit register (via angle embedding).
    - Applies a parameterized sequence of single-qubit rotations and entangling gates for several layers.
    - Measures expectation values of Pauli-Z on each qubit, returning a 4-dimensional output.

- **`QuantumLayer` module**
  - Wraps the QNode into a PyTorch `nn.Module`.
  - Holds the trainable quantum parameters as a `nn.Parameter` tensor.
  - In `forward(x_batch)`:
    - Loops over the batch items.
    - Feeds each 4-dimensional input vector into the QNode.
    - Stacks the 4-dimensional outputs into a `[batch_size, 4]` tensor.
    - Casts to `float32` for compatibility with other layers.

- **Hybrid head**
  - Classical projection: a small layer mapping the 512-dimensional backbone feature to 4 dimensions (for example, `Linear(512, 4)` with an activation).
  - Quantum layer: the `QuantumLayer` defined above.
  - Final head: a small linear layer mapping the 4-dimensional quantum output to 1 or 2 logits, depending on binary vs multi-class tasks.

- **Training loop**
  - Uses the same aligned face dataloaders as the classical baseline.
  - Loss: usually `CrossEntropyLoss` for the final logits.
  - Optimizer: `Adam` or `AdamW` over the projection, quantum layer, and head parameters.
  - Tracks training and validation loss and accuracy each epoch.
  - Saves the best-performing hybrid model checkpoint (for example, `hybrid_best.pt`).

---

### `ensemble.ipynb`

Trains an ensemble of clean hybrid experts.

**High-level flow**

1. Configure the number of experts and training hyperparameters.
2. For each expert:
   - Build a hybrid model with potentially different quantum-circuit depth or entangling pattern.
   - Optionally sample a bootstrap subset of the training data.
   - Train and save the expert’s checkpoint.
3. Optionally, precompute and cache features to avoid repeated backbone passes.

**Main components**

- **Configuration**
  - Defines:
    - `N_EXPERTS`: number of hybrid models in the ensemble.
    - Random seeds per expert.
    - Shared hyperparameters (batch size, learning rate, number of epochs).
    - Paths for saving checkpoints and logs.

- **Feature caching (optional)**
  - Precomputes the 512-dimensional (or 4-dimensional) features for all training and validation images using the frozen ResNet backbone.
  - Stores them in memory or on disk to speed up repeated training of experts.

- **Expert model constructor**
  - A helper that builds one hybrid model:
    - Takes in an expert-specific configuration (e.g., quantum depth, entangling layout).
    - Returns a `HybridModel` instance (backbone + projection + quantum circuit + head).

- **Single-expert training function**
  - A wrapper similar to the training loop in `hybrid.ipynb`:
    - Accepts an `expert_id` and its configuration.
    - Uses either raw images or cached features.
    - Trains for a fixed number of epochs with early stopping.
    - Saves the model state to `artifacts/ensemble/model_{expert_id}.pt`.

- **Ensemble driver**
  - A loop over `expert_id` in `0..N_EXPERTS-1` that:
    - Constructs the expert.
    - Trains it via the single-expert function.
    - Records the path of the saved checkpoint for later evaluation.

---

### `noise.ipynb`

Trains a single noise-aware hybrid model by inserting quantum noise into the VQC.

**High-level flow**

1. Define noise channels and a noisy quantum device (for example, using `default.mixed` in PennyLane).
2. Build a hybrid model structurally similar to `hybrid.ipynb` but with noise in the quantum circuit.
3. Train and evaluate the noisy hybrid and compare it to a clean one.

**Main components**

- **Noisy device and channels**
  - Sets up a quantum device that supports mixed states and noise.
  - Defines noise types such as:
    - T1-like (amplitude damping).
    - T2-like (dephasing).
    - Small random over-rotations on rotation gates.
  - Adds these channels after or between rotation / entangling gates inside each circuit layer.

- **Noisy quantum circuit definition**
  - Similar to the clean VQC, but with explicit noise operations.
  - Still maps a 4-dimensional classical input to a 4-dimensional expectation-value output.

- **Hybrid model with noise**
  - Uses the same backbone and projector as `hybrid.ipynb`.
  - Replaces the clean `QuantumLayer` with a noisy version that calls the noisy QNode.

- **Training and comparison**
  - Trains the noisy hybrid on the verification dataset.
  - Optionally reuses the clean hybrid training code to facilitate direct comparison.
  - Records validation accuracy for several noise-strength settings to study robustness.

**How to Run**

- **Step 1: Configuration & Training**
  - Locate the cell comment: `# QuantumLayer`
  - Uncomment the noise type you want (e.g., `noise = 'T12'`)
  - Run the cell to train.

- **Step 2: Evaluation**
  - Continue executing the subsequent cells in order until reach the specific cell containing the matching comment (e.g., # T12)
  - Run this cell to see the final accuracy.
---

### `ensemble_noise.ipynb`

Trains an ensemble of noisy hybrid experts.

**High-level flow**

1. Define a function that trains a single noisy hybrid expert, given:
   - Expert index.
   - Noise type and strength.
   - Quantum-circuit depth and structure.
2. Loop over a fixed number of experts.
3. Save each noisy expert’s checkpoint for later ensemble prediction.

**Main components**

- **Noise configuration**
  - Chooses one or more noise types, such as:
    - `"T1"`, `"T2"`, or `"rotation"`.
    - Composite types like `"T12r"` to combine several channels.
  - Associates each expert with a particular noise configuration or a small random perturbation of it.

- **`train_noise_expert(...)` function**
  - Constructs the noisy hybrid model using the chosen noise configuration.
  - Loads the backbone checkpoint (from `classical.ipynb`).
  - Trains for a specified number of epochs and batch size.
  - Saves a checkpoint like `hybrid_noise_T2_expert{expert_id}.pt`.

- **Ensemble training driver**
  - A loop over `expert_id`:
    - Calls `train_noise_expert(expert_id, ...)`.
    - Collects the checkpoint paths.
  - Used to generate sets of noisy experts for evaluation.

---

### `predict.ipynb`

Runs evaluation for a single model (classical or hybrid) on a held-out test set.

**High-level flow**

1. Build a test dataloader from a directory of images arranged in an ImageFolder structure.
2. Load a chosen model checkpoint.
3. Run the model on the test data and compute summary metrics.

**Main components**

- **Test data loader**
  - Reads images from a root directory where subfolders correspond to labels.
  - Applies the same resizing and normalization as used in training.

- **Model loaders**
  - Helper functions to:
    - Construct a ResNet-18 model with the same head shape and load a classical checkpoint.
    - Construct a hybrid model (backbone + quantum head) and load a hybrid checkpoint.

- **Evaluation loop**
  - Iterates over the test dataloader.
  - For each batch:
    - Runs the model in evaluation mode under `torch.no_grad()`.
    - Computes predicted labels and optionally probabilities.
  - Aggregates:
    - Total correct predictions.
    - Overall accuracy.
    - Optionally, per-class statistics or confidence histograms.

---

### `ensemble_predict.ipynb`

Evaluates ensembles and compares them with single models.

**High-level flow**

1. Load:
   - A classical baseline model.
   - One or more single hybrid models.
   - Sets of ensemble experts (clean and/or noisy).
2. For each test image, run multiple prediction strategies.
3. Compare their performance metrics.

**Main components**

- **Model loading**
  - Loads all required checkpoints:
    - Classical baseline.
    - Single hybrid model.
    - Clean hybrid ensemble experts.
    - Noisy hybrid ensemble experts.
  - Optionally loads a meta-learner (stacking model) if used.

- **Ensemble prediction helpers**
  - Functions that, for one image or batch:
    - Run each expert and collect logits or probabilities.
    - Aggregate predictions by:
      - Simple average (soft voting).
      - Weighted average if temperatures or weights are learned.
      - Passing concatenated expert outputs into a meta-classifier.

- **Evaluation routine**
  - Iterates over the test dataloader.
  - For each batch, computes predictions using:
    - Classical baseline.
    - Single hybrid model.
    - Clean ensemble.
    - Noisy ensemble (if applicable).
  - Tracks:
    - Accuracy for each method.
    - Simple robustness indicators (for example, variance across experts).
  - Prints a compact comparison summary at the end.

---

## Typical Workflow

1. **Preprocess faces**  
   Use `preprocess.py` to generate aligned 224×224 face crops in an ImageFolder structure.

2. **Train the classical baseline**  
   Run `classical.ipynb` to train ResNet-18 and export backbone and full-model checkpoints.

3. **Train hybrid models and ensembles**  
   - `hybrid.ipynb`: single clean hybrid model.  
   - `ensemble.ipynb`: ensemble of clean hybrid experts.  
   - `noise.ipynb`: single noisy hybrid model.  
   - `ensemble_noise.ipynb`: ensemble of noisy hybrid experts.

4. **Evaluate**  
   - `predict.ipynb`: evaluate any single model on the test set.  
   - `ensemble_predict.ipynb`: evaluate ensembles and compare them with the classical and single-hybrid baselines.

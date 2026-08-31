# Motor Imagery EEG Classification

This project investigates the **cross-subject generalization of motor imagery EEG classification models**. The objective is to evaluate whether models trained on EEG recordings from a group of participants can classify motor imagery from a participant who was completely unseen during model development.

## Dataset

Experiments use the **BNCI2014-001** motor imagery dataset, accessed through [MOABB](https://moabb.neurotechx.com/).

The dataset contains recordings from **9 subjects** performing four motor imagery tasks:

* Left hand
* Right hand
* Both feet
* Tongue

EEG was recorded using **22 EEG channels** at a sampling frequency of **250 Hz**.

The data are loaded using MOABB's MotorImagery paradigm. Subject, session, and run information provided by MOABB is retained alongside each EEG trial.

## Cross-Subject Evaluation

Model performance is evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation.

For each LOSO fold, one of the nine participants is held out as the test subject. The remaining eight participants form the development set. A validation subject is selected from the development subjects for model selection and early stopping, while the remaining subjects are used for training.

The held-out test participant is not used during model training, preprocessing parameter estimation, model selection, or early stopping.

The procedure is repeated nine times so that every participant serves as the unseen test subject once.

This evaluation is intended to measure subject-independent generalization, rather than performance when training and testing on EEG trials belonging to the same participants.

## Preprocessing

EEG data are represented as trials with dimensions corresponding to channels and time samples.

Any preprocessing operation that estimates parameters from the data is fitted using the training subjects only. The resulting transformation is then applied to the validation and test data.

This prevents information from the held-out participant from influencing model development.

## Classification Models

Four classification approaches are evaluated using the same LOSO protocol.

### CSP-LDA

Common Spatial Patterns (CSP) with Linear Discriminant Analysis (LDA) is included as a classical motor imagery EEG baseline.

CSP spatial filters are estimated exclusively from the training subjects before being applied to validation and test trials.

### EEGNet

EEGNet is a compact convolutional neural network developed specifically for EEG-based brain-computer interface applications. It uses temporal and depthwise spatial convolutions to learn spectral and spatial representations directly from EEG trials.

### ShallowConvNet

ShallowConvNet is included as an additional EEG-specific convolutional architecture. Its design emphasizes temporal filtering followed by spatial filtering and transformations related to band-power feature extraction.

### ATCNet

ATCNet combines convolutional EEG feature extraction with attention and temporal convolutional components. It is included as a more complex deep-learning approach for modeling temporal information in motor imagery EEG.

## Model Training

The neural network models are trained using the training subjects within each LOSO fold.

Model selection and early stopping are based on performance on the validation subject. Once training is complete, the selected model is evaluated on the held-out test subject.

The test subject is used only for final evaluation.

Random seeds are fixed and recorded to allow repeated experiments and assessment of variability in neural network training.

## Evaluation Metrics

Performance is measured separately for every held-out participant using:

* Accuracy
* Balanced accuracy
* Macro F1-score
* Cohen's kappa

Confusion matrices and trial-level predictions are also retained for subsequent analysis.

Final cross-subject performance is summarized across the nine held-out subjects.

## Project Structure

The repository is organized into:

* **Data** — dataset information and acquisition details
* **Code** — data loading, preprocessing, LOSO splitting, models, training, and evaluation
* **Config** — model and training configurations
* **Results** — model predictions, checkpoints, and clean LOSO results
* **Archive** — previous experimental notebooks

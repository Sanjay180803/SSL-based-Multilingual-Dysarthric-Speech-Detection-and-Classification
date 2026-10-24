# Original Google Colab notebook export, kept for reference. It is not
# runnable as a .py file (it uses "!" shell commands and "cd"). The
# maintained pipeline is in src/dysarthria/.

# -*- coding: utf-8 -*-
"""Hubert Extraction"""

# Clone repo for DistilALHuBERT
!git clone https://github.com/backspacetg/distilAlhubert.git

from google.colab import drive
drive.mount('/content/drive')  # If using Google Drive

drive.mount("/content/drive", force_remount=True)

# Change directory to repo
cd distilAlhubert/

!pip install -r /path/to/your/distilAlhubert/requirements.txt
!pip install s3prl

import torch
from src.upstream.alhubert.expert import UpstreamExpert

# Path to your HuBERT model checkpoint
model_ckpt_path = "/path/to/your/model_checkpoint.ckpt"

model = UpstreamExpert(model_ckpt_path)
data = [torch.randn(10000) for _ in range(2)]  # Example 16KHz data
states = model(data)
print(states["last_hidden_state"].shape)
print(len(states["hidden_states"]))

import os
import shutil
import torchaudio

# Paths
input_folder = "/path/to/your/input_audio_dataset"  # Folder with .wav files
output_folder = "/path/to/your/output_embeddings"  # Folder to save embeddings
processed_structure_folder = "/path/to/your/processed_audio_structure"  # Folder to move processed .wav files

# Create necessary directories
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_structure_folder, exist_ok=True)

def create_structure(input_folder, output_folder):
    """Replicates the folder structure of input_folder in output_folder."""
    for subdir, _, _ in os.walk(input_folder):
        relative_subdir = os.path.relpath(subdir, input_folder)
        output_subdir = os.path.join(output_folder, relative_subdir)
        os.makedirs(output_subdir, exist_ok=True)

def process_audio_file(audio_path):
    """Processes a single audio file to extract embeddings."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        data = [waveform.squeeze(0)]
        with torch.no_grad():
            states = model(data)
        return states["last_hidden_state"]
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def safe_move(src, dst):
    """Safely moves a file, handling cross-device issues."""
    try:
        shutil.move(src, dst)
    except OSError as e:
        if "Invalid cross-device link" in str(e):
            try:
                shutil.copy(src, dst)
                os.remove(src)
            except Exception as copy_err:
                print(f"Failed to copy/move {src} to {dst}: {copy_err}")
        else:
            print(f"Error moving {src} to {dst}: {e}")

def process_folder(input_folder, output_folder, processed_structure_folder):
    """Processes all .wav files in the folder, saves embeddings, and replicates folder structure."""
    create_structure(input_folder, processed_structure_folder)
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.startswith('.'):
                continue
            if file.endswith(".wav"):
                audio_path = os.path.join(subdir, file)
                try:
                    features = process_audio_file(audio_path)
                    if features is None:
                        continue
                    relative_subdir = os.path.relpath(subdir, input_folder)
                    output_subdir = os.path.join(output_folder, relative_subdir)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file.replace(".wav", "_embeddings.pt"))
                    torch.save(features, output_file)
                    target_subdir = os.path.join(processed_structure_folder, relative_subdir)
                    os.makedirs(target_subdir, exist_ok=True)
                    target_file = os.path.join(target_subdir, file)
                    safe_move(audio_path, target_file)
                    print(f"Processed {audio_path}, saved embeddings to {output_file}, and moved .wav to {target_file}")
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

# Optional: copy dataset locally for faster processing
local_data_folder = "/path/to/your/local_copy_of_dataset"
try:
    shutil.copytree(input_folder, local_data_folder)
    input_folder = local_data_folder
except Exception as e:
    print(f"Failed to copy data locally: {e}. Proceeding with original input folder.")

# Process the folder
process_folder(input_folder, output_folder, processed_structure_folder)

# Example: Load a processed embedding
pt_file_path = "/path/to/your/embeddings_folder/example_embeddings.pt"
embeddings = torch.load(pt_file_path)
print("Shape of the embeddings:", embeddings.shape)
print("Embeddings:", embeddings)

# Classification script setup
embedding_folder = "/path/to/your/embeddings_folder"

folder_to_label = {
    "subfolder1": 0,
    "subfolder2": 1,
    "subfolder3": 2,
    "subfolder4": 3
}

all_embeddings, all_labels = [], []
fixed_size = 768

for subdir, dirs, files in os.walk(embedding_folder):
    for file in files:
        if file.endswith(".pt"):
            pt_file_path = os.path.join(subdir, file)
            embeddings = torch.load(pt_file_path, map_location="cpu", weights_only=True).flatten().numpy()
            embeddings = embeddings[:fixed_size] if len(embeddings) > fixed_size else np.pad(embeddings, (0, fixed_size - len(embeddings)))
            all_embeddings.append(embeddings)
            folder_name = os.path.basename(subdir)
            label = folder_to_label.get(folder_name, -1)
            all_labels.append(label)

all_embeddings = np.array(all_embeddings)
all_labels = np.array(all_labels)
print("Embeddings shape:", all_embeddings.shape)
print("Labels shape:", all_labels.shape)

## Replace this portion with the CNN Model definition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=42)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM classifier: {accuracy * 100:.2f}%")

# Rename files example
def rename_files_in_subfolder(folder_path, suffix):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_extension = os.path.splitext(file)
            new_file_name = file_name + suffix + file_extension
            new_file_path = os.path.join(root, new_file_name)
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')

subfolder_1 = '/path/to/your/features_folder/severe'
subfolder_2 = '/path/to/your/features_folder/mild'
subfolder_3 = '/path/to/your/features_folder/moderate'
subfolder_M_control = '/path/to/your/features_folder/M_control'
subfolder_F_control = '/path/to/your/features_folder/F_control'

rename_files_in_subfolder(subfolder_1, '_dysarthria')
rename_files_in_subfolder(subfolder_2, '_dysarthria')
rename_files_in_subfolder(subfolder_3, '_dysarthria')
rename_files_in_subfolder(subfolder_M_control, '_control')
rename_files_in_subfolder(subfolder_F_control, '_control')

# Averaging embeddings example
def average_and_save_embeddings(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        save_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(save_subfolder, exist_ok=True)
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                embedding = torch.load(file_path).numpy()
                averaged_embedding = np.mean(embedding, axis=1)
                save_file_path = os.path.join(save_subfolder, file.replace('.pt', '_averaged.npy'))
                np.save(save_file_path, averaged_embedding)
                print(f"Averaged embedding saved at: {save_file_path}")

input_folder = '/path/to/your/features_folder'
output_folder = '/path/to/your/averaged_features_folder'

average_and_save_embeddings(input_folder, output_folder)

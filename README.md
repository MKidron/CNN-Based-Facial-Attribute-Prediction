# M-KIDRON
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import time

# Set multiprocessing start method for Windows
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

# Data Preparation
data_path = r"C:\Users\M KIDRON\OneDrive\Desktop\mlprojects\first\age__gender.csv"
try:
    face_data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please verify the file path.")

# Print initial dataset size
print(f"Number of rows: {len(face_data)}")

# Preprocess pixels and filter for 48x48 images
face_data['pixels'] = face_data['pixels'].fillna('').apply(lambda x: np.fromstring(x, sep=' ', dtype=np.uint8))
face_data = face_data[face_data['pixels'].apply(lambda x: len(x) == 2304)].copy()

# Check if dataset is empty after filtering
if face_data.empty:
    raise ValueError("No valid data after filtering. Ensure the dataset contains valid 48x48 pixel data.")
print(f"Rows after pixel filtering: {len(face_data)}")

# Bin ages into 10 categories and encode labels
face_data['age_category'] = pd.cut(face_data['age'], bins=10, labels=False, include_lowest=True)
face_data['gender_label'] = face_data['gender'].astype(int)
face_data['ethnicity_label'] = pd.Categorical(face_data['ethnicity']).codes

# Verify number of age bins for model compatibility
num_age_bins = face_data['age_category'].nunique()
print(f"Number of age bins: {num_age_bins}")

# Split data into train (70%), validation (15%), and test (15%)
train_data, temp_data = train_test_split(face_data, test_size=0.3, random_state=789)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=789)
print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

# Check index ranges for debugging
print(f"Train index range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Val index range: {val_data.index.min()} to {val_data.index.max()}")
print(f"Test index range: {test_data.index.min()} to {test_data.index.max()}")

# Custom Dataset
class FacialFeaturesDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            pixels = row['pixels']
            if not isinstance(pixels, np.ndarray) or pixels.size != 2304:
                raise ValueError(f"Invalid pixel data at index {idx}")
            # Keep image as NumPy array for transforms
            image = pixels.astype(np.float32).reshape(48, 48, 1)
            if self.transform:
                image = self.transform(image)
            gender = torch.tensor(row['gender_label'], dtype=torch.float32)
            age = torch.tensor(row['age_category'], dtype=torch.long)
            ethnicity = torch.tensor(row['ethnicity_label'], dtype=torch.long)
            return image, gender, age, ethnicity
        except Exception as e:
            print(f"Error in dataset at index {idx}: {str(e)}")
            raise

# Define transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),  # Converts NumPy array to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create datasets
train_dataset = FacialFeaturesDataset(train_data, transform=train_transforms)
val_dataset = FacialFeaturesDataset(val_data, transform=val_test_transforms)
test_dataset = FacialFeaturesDataset(test_data, transform=val_test_transforms)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Verify DataLoader
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# Test DataLoader
print("Testing train_loader...")
try:
    for i, (images, gender, age, ethnicity) in enumerate(train_loader):
        print(f"Batch {i}: Image shape: {images.shape}, Gender shape: {gender.shape}, Age shape: {age.shape}, Ethnicity shape: {ethnicity.shape}")
        if i > 50:  # Limit to avoid long runs
            break
    print("DataLoader test completed.")
except Exception as e:
    print(f"DataLoader test failed: {str(e)}")
    raise

# CNN Model
class FacialAttributeCNN(nn.Module):
    def __init__(self, num_age_groups=10, num_ethnicities=5, dropout_rate=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        feature_size = 64 * 4 * 4
        self.gender_fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        self.age_fc = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_age_groups)
        )
        self.ethnicity_fc = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_ethnicities)
        )

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.flatten(1)
        gender_out = self.gender_fc(features).squeeze(1)
        age_out = self.age_fc(features)
        ethnicity_out = self.ethnicity_fc(features)
        return gender_out, age_out, ethnicity_out

if __name__ == '__main__':
    # Initialize model
    model = FacialAttributeCNN(num_age_groups=num_age_bins, num_ethnicities=5)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss functions
    gender_loss_fn = nn.BCEWithLogitsLoss()
    age_loss_fn = nn.CrossEntropyLoss()
    ethnicity_loss_fn = nn.CrossEntropyLoss()

    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Loss weights
    gender_weight = 0.5
    age_weight = 1.0
    ethnicity_weight = 1.0

    # Training and validation loop
    num_epochs = 10
    best_val_loss = float('inf')
    best_model_path = "best_facial_attribute_model.pth"
    patience = 5
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_gender_correct = 0
        train_age_correct = 0
        train_ethnicity_correct = 0
        train_total = 0

        try:
            for batch_idx, (images, gender, age, ethnicity) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                images, gender, age, ethnicity = images.to(device), gender.to(device), age.to(device), ethnicity.to(device)
                
                optimizer.zero_grad()
                gender_out, age_out, ethnicity_out = model(images)
                
                # Verify output shapes
                if batch_idx == 0:
                    print(f"Gender out shape: {gender_out.shape}, Expected: {gender.shape}")
                    print(f"Age out shape: {age_out.shape}, Expected: {age.shape}")
                    print(f"Ethnicity out shape: {ethnicity_out.shape}, Expected: {ethnicity.shape}")

                gender_loss = gender_loss_fn(gender_out, gender) * gender_weight
                age_loss = age_loss_fn(age_out, age) * age_weight
                ethnicity_loss = ethnicity_loss_fn(ethnicity_out, ethnicity) * ethnicity_weight
                total_loss = gender_loss + age_loss + ethnicity_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                gender_preds = (torch.sigmoid(gender_out) > 0.5).float()
                train_gender_correct += (gender_preds == gender).sum().item()
                train_age_correct += (torch.argmax(age_out, dim=1) == age).sum().item()
                train_ethnicity_correct += (torch.argmax(ethnicity_out, dim=1) == ethnicity).sum().item()
                train_total += images.size(0)

                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")

        except Exception as e:
            print(f"Error in epoch {epoch+1}, batch {batch_idx}: {str(e)}")
            raise

        train_loss /= len(train_loader)
        train_gender_acc = train_gender_correct / train_total
        train_age_acc = train_age_correct / train_total
        train_ethnicity_acc = train_ethnicity_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_gender_correct = 0
        val_age_correct = 0
        val_ethnicity_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, gender, age, ethnicity in val_loader:
                images, gender, age, ethnicity = images.to(device), gender.to(device), age.to(device), ethnicity.to(device)
                
                gender_out, age_out, ethnicity_out = model(images)
                
                gender_loss = gender_loss_fn(gender_out, gender) * gender_weight
                age_loss = age_loss_fn(age_out, age) * age_weight
                ethnicity_loss = ethnicity_loss_fn(ethnicity_out, ethnicity) * ethnicity_weight
                total_loss = gender_loss + age_loss + ethnicity_loss
                
                val_loss += total_loss.item()
                gender_preds = (torch.sigmoid(gender_out) > 0.5).float()
                val_gender_correct += (gender_preds == gender).sum().item()
                val_age_correct += (torch.argmax(age_out, dim=1) == age).sum().item()
                val_ethnicity_correct += (torch.argmax(ethnicity_out, dim=1) == ethnicity).sum().item()
                val_total += images.size(0)

        val_loss /= len(val_loader)
        val_gender_acc = val_gender_correct / val_total
        val_age_acc = val_age_correct / val_total
        val_ethnicity_acc = val_ethnicity_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Gender Acc: {train_gender_acc:.4f}, Age Acc: {train_age_acc:.4f}, Ethnicity Acc: {train_ethnicity_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Gender Acc: {val_gender_acc:.4f}, Age Acc: {val_age_acc:.4f}, Ethnicity Acc: {val_ethnicity_acc:.4f}")

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model with validation loss: {best_val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # Camera Interface for Gender, Age, and Ethnicity Prediction (Runs for 6 seconds)
    print("Starting camera interface for facial attribute prediction...")
    model.eval()
    inference_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    gender_labels = {0: "Male", 1: "Female"}
    age_labels = {i: f"Age Group {i+1}" for i in range(num_age_bins)}
    ethnicity_labels = {i: cat for i, cat in enumerate(pd.Categorical(face_data['ethnicity']).categories)}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Failed to load face cascade classifier.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Failed to open webcam.")

    start_time = time.time()
    duration = 10  # Run for 6 seconds
    predictions = []

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
            face = face.astype(np.float32)[:, :, np.newaxis]
            face_tensor = inference_transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                gender_out, age_out, ethnicity_out = model(face_tensor)
                gender_pred = (torch.sigmoid(gender_out) > 0.5).float().item()
                age_pred = torch.argmax(age_out, dim=1).item()
                ethnicity_pred = torch.argmax(ethnicity_out, dim=1).item()

                gender_label = gender_labels[int(gender_pred)]
                age_label = age_labels[age_pred]
                ethnicity_label = ethnicity_labels[ethnicity_pred]

                # Store prediction
                predictions.append(f"{gender_label}, {age_label}, {ethnicity_label}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{gender_label}, {age_label}, {ethnicity_label}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Facial Attribute Prediction', frame)
        cv2.waitKey(1)  # Brief delay to display frame

    # Print results and terminate
    cap.release()
    cv2.destroyAllWindows()
    print("Camera interface closed after 6 seconds.")
    if predictions:
        print("Predictions:")
        for i, pred in enumerate(set(predictions), 1):  # Use set to avoid duplicate predictions
            print(f"Face {i}: {pred}")
    else:
        print("No faces detected.")
    print("Program terminated.")

    #Output 
      Train Loss: 1.4318, Gender Acc: 0.9192, Age Acc: 0.6301, Ethnicity Acc: 0.8446
  Val Loss: 1.7721, Gender Acc: 0.8954, Age Acc: 0.5720, Ethnicity Acc: 0.7773
  Saved best model with validation loss: 1.7721
Starting camera interface for facial attribute prediction...

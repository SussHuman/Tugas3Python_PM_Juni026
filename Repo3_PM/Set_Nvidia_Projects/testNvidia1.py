#torch
import torch
import torch.nn as nn
from torchvision import models

# nvidia
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# ==========================================
# 1. NVIDIA AUGMENTATION & DALI PIPELINE
# ==========================================
@pipeline_def
def create_dali_pipeline(data_dir, batch_size, num_threads, device_id):
    # membaca gambar
    jpegs, labels = fn.readers.file(file_root=data_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs, device="mixed")
    
    # Augmentasi NVIDIA
    images = fn.resize(images, resize_x=256, resize_y=256)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(probability=0.5)
    )
    
    return images, labels.gpu()

# ==========================================
# 2. PRE-TRAIN NVIDIA & CNN SETUP
# ==========================================
def get_pretrained_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Pindahkan model ke GPU NVIDIA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    batch_size = 32
    num_classes = 10 
    
    # Inisialisasi DALI Pipeline
    pipe = create_dali_pipeline(batch_size=batch_size, num_threads=4, device_id=0, data_dir="/path/ke/dataset/anda")
    pipe.build()
    
    # Mengubah DALI Pipeline
    train_loader = DALIGenericIterator([pipe], ['data', 'label'], reader_name='Reader')
    
    # Setup Model
    model, device = get_pretrained_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    print("Mulai proses training di NVIDIA GPU...")
    model.train()
    
    epochs = 5
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs = data[0]['data']
            labels = data[0]['label'].squeeze(-1).long()
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")
        
        # Reset DALI iterator
        train_loader.reset()
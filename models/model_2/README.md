
# Emotion Recognition Model

This project involves building and training an emotion recognition model using PyTorch. The model is designed to classify images of human faces into different emotion categories. The project uses a custom dataset for training and evaluation.

## Project Structure

- `main.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.

## Requirements

To run this project, you need the following libraries:

- Python 3.x
- PyTorch
- Torchvision
- Scikit-learn

You can install the required libraries using the following command:

```bash
pip install torch torchvision scikit-learn
```

## Dataset

The dataset should be organized into the following directory structure:

```
fer_dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
```

Replace `class_1`, `class_2`, etc., with the actual class names for emotions.

## Data Preprocessing

The images are converted to grayscale and normalized. Data augmentation techniques such as random horizontal flip and random rotation are applied to the training set.

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}
```

## Model Training

The model is based on ResNet18, with modifications to suit the emotion recognition task. The training process involves the following steps:

1. Load the dataset using `ImageFolder` and `DataLoader`.
2. Define the model, loss function, optimizer, and learning rate scheduler.
3. Train the model for a specified number of epochs.
4. Save the best model based on the validation F1-Score.

```python
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
torch.save(model.state_dict(), 'emotion_recognition_model.pth')
```

## Evaluation

The model's performance is evaluated using precision, recall, and F1-Score metrics.

```python
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1-Score: {f1_score:.4f}')
```

## Results

The final model achieved the following performance on the test set:

- **Precision**: 0.6463
- **Recall**: 0.6495
- **F1-Score**: 0.6448
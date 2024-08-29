import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import os, glob, math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

class SentimentDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_length=1000):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = self.statements[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(statement)
        if len(tokens) > self.max_length:
            i_start = torch.randint(low=0, high=len(tokens) - self.max_length + 1, size=(1, )).item()
            tokens = tokens[i_start:i_start+self.max_length]
        tokens = torch.tensor(tokens)

        return tokens, torch.tensor(label)

def collate_fn(batch):
    tokens, labels = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return tokens_padded, labels


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_classes):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        output = self.fc(output)
        return output


# Assume you have a tokenizer function
def simple_tokenizer(text):
    return [encoder[c] for c in text]  # Simple example: convert each character to its ASCII value



if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    file = glob.glob(os.path.expanduser("~/Documents/projects/chatgpt-from-scratch/data/*.csv"))[0]
    df = pd.read_csv(file, index_col=0).dropna(how="any", axis=0)
    
    max_length = int(df["statement"].apply(len).quantile(0.6))
    
    temp_set = set()
    for item in df["statement"].apply(set):
        temp_set = temp_set | item
   
    vocab_size = len(temp_set)
    
    encoder = {s: i for i, s in enumerate(temp_set)}
    
    statements = df["statement"].values
    labels = df["status"].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    train_statements, val_statements, train_labels, val_labels = train_test_split(statements, encoded_labels, test_size=0.2, random_state=42)
    # Create datasets
    train_dataset = SentimentDataset(train_statements, train_labels, tokenizer=simple_tokenizer, max_length=max_length)
    val_dataset = SentimentDataset(val_statements, val_labels, tokenizer=simple_tokenizer, max_length=max_length)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    
    # Instantiate the model
    model = CustomTransformerModel(vocab_size=vocab_size, d_model=256, nhead=8, num_encoder_layers=6, num_classes=len(label_encoder.classes_)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        print(f"start training epoch {epoch}...")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Optional: Evaluate on the validation set after each epoch
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)        
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}, Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')
        model.train()
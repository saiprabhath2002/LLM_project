import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available else "cpu"

# Datasets and dataloaders
ds = load_dataset("facebook/xnli", "all_languages")

num_samples = 10000
random_indices = torch.randperm(len(ds['train']))[:num_samples].tolist()
train_dataset = Subset(ds['train'], random_indices)

xnli_lang_list = train_dataset[0]['hypothesis']['language']

num_samples = 500
random_indices = torch.randperm(len(ds['validation']))[:num_samples].tolist()
val_dataset = Subset(ds['validation'], random_indices)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")


def collate_fn(batch, lang_id):
    sentences = []
    for sample in batch:
        sen1 = list(sample['premise'].values())[lang_id]
        sen2 = sample['hypothesis']['translation'][lang_id]
        # print(sen1)
        # print(sen2)
        # print(type(sen1_list))
        # print(sen2_list)
        sen = f"premise: {sen1} hypothesis: {sen2}"
        sentences.append(sen)

    # print(sentences)
    sentence_embeddings = tokenizer(sentences, return_tensors="pt", padding=True)
    # print(sentence_embeddings)
    
    labels = torch.tensor([sample['label'] for sample in batch])

    return sentence_embeddings, labels
    
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda batch: collate_fn(batch, 4))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda batch: collate_fn(batch, 4))



# Model
class ModifiedFullModel(nn.Module):
    def __init__(self, input_dim=2048, num_heads=8, num_layers=6, hidden_dim=2048, num_classes=3):
        super(ModifiedFullModel, self).__init__()
        self.bloom_model = bmodel
        self.bloom_model = self.bloom_model.transformer
        
        for param in self.bloom_model.parameters():
            param.requires_grad = False

        self.attention_head = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        
        self.classification_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )


    def forward(self, x, device, labels=None):
        
        batch_size = x['input_ids'].shape[0]
        # print(batch_size)
        
        # return
        bloom_out = self.bloom_model(**x).last_hidden_state
        cls_token = torch.zeros(batch_size, 1, 2048).to(device)
        x = torch.cat((cls_token, bloom_out), dim=1)
        x = x.transpose(0,1)
        x, _ = self.attention_head(x, x, x)
        x = x.transpose(0,1)
        # cls_output = x[0]
        output = self.classification_mlp(x[:,0,:])
        
        return output
        

config = AutoConfig.from_pretrained("bigscience/bloom-1b7")
config.output_hidden_states = True 
bmodel = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", config=config)

model = ModifiedFullModel().to(device)
##training

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.CrossEntropyLoss()

num_epochs = 60

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_batches = len(train_loader)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_loader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        # print(inputs[0].shape)

        optimizer.zero_grad()
        outputs = model(inputs, device)
        # print(len(outputs))
        # print(outputs.shape)
        # print(labels)
        # break
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        del inputs, labels, outputs
    # break
    avg_train_loss = total_train_loss / total_batches
    training_losses.append(avg_train_loss)

    # Validation process
    model.eval()
    
    total_val_loss = 0
    correct_counts = 0
    layer_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs, device)
            
            val_loss = loss_function(outputs, labels).item()
            total_val_loss += val_loss
            _, predicted = torch.max(outputs, 1)
            correct_counts += (predicted == labels).sum().item()
            layer_samples += labels.size(0)

            del inputs, labels, outputs
                
    if layer_samples > 0:
        epoch_val_accuracy = correct_counts / layer_samples
        validation_accuracies.append(epoch_val_accuracy)

        epoch_val_loss = total_val_loss / layer_samples
        validation_losses.append(epoch_val_loss)
        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')


# Saving model and training metrics
model_save_path = 'task3_model.pth'
torch.save(model.state_dict(), model_save_path)

df = pd.DataFrame({"train_loss":training_losses, "val_loss":validation_losses, "val_acc":validation_accuracies})
df.to_csv('task3_train_metrics.csv')


# Training loss plot
plt.figure(figsize=(7, 5))
plt.plot(training_losses, label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_plot.png')
# plt.show()
plt.close()

# Validation Accuracy Plot
plt.plot(validation_accuracies)
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('validation_accuracy_plot.png')
# plt.show()
plt.close()

# Validation Loss Plot
plt.plot(validation_losses)
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('validation_loss_plot.png')
# plt.show()
plt.close()


# Test on low resource languages

model.eval()
test_accuracies = {}

for i in range(14):
    test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda batch: collate_fn(batch, i))
    correct_counts = 0
    layer_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs, device)
            
            _, predicted = torch.max(outputs, 1)
            correct_counts += (predicted == labels).sum().item()
            layer_samples += labels.size(0)
    
            del inputs, labels, outputs
                
    if layer_samples > 0:
        epoch_val_accuracy = correct_counts / layer_samples
        test_accuracies[xnli_lang_list[i]] = epoch_val_accuracy
        print(f'{xnli_lang_list[i]} Accuracy: {epoch_val_accuracy:.4f}')

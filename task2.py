import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig


ds = load_dataset("facebook/xnli", "all_languages")
num_samples = 10000
random_indices = torch.randperm(len(ds['train']))[:num_samples].tolist()
train_dataset = Subset(ds['train'], random_indices)

num_samples = 500
random_indices = torch.randperm(len(ds['validation']))[:num_samples].tolist()
val_dataset = Subset(ds['validation'], random_indices)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")



def collate_fn(batch):
    sentences = []
    for sample in batch:
        sen1_list = list(sample['premise'].values())
        sen2_list = sample['hypothesis']['translation']
        # sen1 = sen1_list[random.randint(0, len(sen1_list)-1)]
        # sen2 = sen2_list[random.randint(0, len(sen2_list)-1)]
        # print(sen1)
        # print(sen2)
        # print(type(sen1_list))
        # print(sen2_list)
        sen1 = random.choice(sen1_list)
        sen2 = random.choice(sen2_list)
        sen = f"premise: {sen1} hypothesis: {sen2}"
        sentences.append(sen)

    sentence_embeddings = tokenizer(sentences, return_tensors="pt", padding=True)
    # print(sentence_embeddings)
    
    labels = torch.tensor([sample['label'] for sample in batch])

    return sentence_embeddings, labels

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)



config = AutoConfig.from_pretrained("bigscience/bloom-1b7")
config.output_hidden_states = True  
bmodel = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", config=config)


device = "cuda" if torch.cuda.is_available else "cpu"

class ModifiedFullModel(nn.Module):
    def __init__(self, input_dim=2048, num_heads=8, num_layers=6, hidden_dim=2048, num_classes=3):
        super(ModifiedFullModel, self).__init__()
        self.bloom_model = bmodel
        self.bloom_model = self.bloom_model.transformer
        
        for param in self.bloom_model.parameters():
            param.requires_grad = False

        self.attention_heads = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads) for _ in range(24)]
        )
        
        self.classification_mlp = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(24)]
        )

        self.accuracy_tracker = []

    def forward(self, x, device, labels=None):
        
        batch_size = x['input_ids'].shape[0]
        bloom_out = self.bloom_model(**x)
        hidden_states = bloom_out.hidden_states
        all_outputs = []  
        for layer_idx in range(24):
            layer_output = hidden_states[layer_idx + 1] 
            cls_token = torch.zeros(batch_size, 1, 2048).to(device)
            layer_output_with_cls = torch.cat((cls_token, layer_output), dim=1)
            
            cls_token_output, _ = self.attention_heads[layer_idx](layer_output_with_cls.transpose(0, 1), layer_output_with_cls.transpose(0, 1), layer_output_with_cls.transpose(0, 1))
            cls_token_output = cls_token_output.transpose(0, 1)

            classification_output = self.classification_mlp[layer_idx](cls_token_output[:, 0, :]) 
            all_outputs.append(classification_output)
        return all_outputs

model = ModifiedFullModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.CrossEntropyLoss()
num_epochs = 1
best_val_acc = 0
training_losses = []  
validation_accuracies = [[] for _ in range(24)]  
validation_losses = [[] for _ in range(24)]



for epoch in range(num_epochs):
    model.train() 
    total_train_loss = 0
    total_batches = len(train_loader)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_loader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        # print(inputs.shape)
        outputs = model(inputs, device)
        optimizer.zero_grad()
        # print(len(outputs))
        # print(outputs[0].shape)
        # print(labels)
        # break
        layer_losses = [loss_function(output, labels) for output in outputs]  
        loss = sum(layer_losses) / len(layer_losses)  
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    # break
        del inputs,labels,outputs
    avg_train_loss = total_train_loss / total_batches
    training_losses.append(avg_train_loss)  

    model.eval()
    total_val_loss = [0] * 24
    correct_counts = [0] * 24
    layer_samples = [0] * 24
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs, device)

            for i, output in enumerate(outputs):
                val_loss = loss_function(output, labels).item()
                total_val_loss[i] += val_loss

                _, predicted = torch.max(output, 1)
                correct_counts[i] += (predicted == labels).sum().item()
                layer_samples[i] += labels.size(0)
            del inputs,labels,outputs
    for i in range(24):
        if layer_samples[i] > 0:
            epoch_val_accuracy = correct_counts[i] / layer_samples[i]
            validation_accuracies[i].append(epoch_val_accuracy)  

            epoch_val_loss = total_val_loss[i] / layer_samples[i]
            validation_losses[i].append(epoch_val_loss)  

            print(f'Layer {i + 1} - Epoch {epoch + 1}: Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')


model_save_path = 'final_60.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model's state dictionary saved to {model_save_path}")



import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
plt.plot(training_losses, label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_plot_60.png')
plt.close()

plt.figure(figsize=(20, 15))
for i in range(24):
    plt.plot(validation_accuracies[i], label=f'Layer {i+1}')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left',fontsize='small')
plt.tight_layout()
plt.savefig('validation_accuracy_plot_60.png')
plt.close()

plt.figure(figsize=(20, 15))
for i in range(24):
    plt.plot(validation_losses[i], label=f'Layer {i+1}')
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left',fontsize='small')
plt.tight_layout()
plt.savefig('validation_loss_plot_60.png')
plt.close()



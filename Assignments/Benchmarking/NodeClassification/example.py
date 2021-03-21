import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.datasets import PPI
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from gcn import GCN
from graphSAGE import GraphSAGE

# Load dataset
path = './Data'
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

# Define the K-fold Cross Validator
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

results = {}  # for fold results

# Model configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCEWithLogitsLoss()
num_epochs = 1
learning_rate = 0.001

# Start print
print('--------------------------------')
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Dataloader
    train_loader = DataLoader(dataset, batch_size=2, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=2, sampler=test_subsampler)

    # Initialize the model
    model = GraphSAGE(train_dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        total_loss = total_examples = 0

        for i, data in enumerate(train_loader, 0):
            # pass data to device
            data = data.to(device)
            # set gradients to zero
            optimizer.zero_grad()
            # perform forward pass
            output = model(data.x, data.edge_index)
            # compute Loss
            loss = criterion(output, data.y)
            # perform backward pass
            loss.backward()
            # perform optimization
            optimizer.step()
            # loss accumulation
            total_loss += loss.item() * data.num_nodes
            total_examples += data.num_nodes
            if (i+1) % 2 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, total_loss / total_examples))
                total_loss = total_examples = 0

    # Process is complete.
    print('Training process has finished. Saving trained model.')
    # Print about testing
    print('Starting testing')
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)
    # Evaluation for this fold
    ys, preds = [], []
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(test_loader, 0):
            # pass data to device
            data = data.to(device)
            ys.append(data.y)
            # Generate outputs
            out = model(data.x, data.edge_index)
            preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    # Print metrics
    F1_Score = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    Precision = precision_score(y, pred) if pred.sum() > 0 else 0
    Recall = recall_score(y, pred) if pred.sum() > 0 else 0
    print('F1-score for fold %d: %d %%' % (fold, F1_Score))
    print('Precision score for fold %d: %d %%' % (fold, Precision))
    print('Recall score for fold %d: %d %%' % (fold, Recall))
    print('--------------------------------')
    results[fold] = (F1_Score, Precision, Recall)

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
# sum=0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    # sum += value
  # print(f'Average: {sum/len(results.items())} %')

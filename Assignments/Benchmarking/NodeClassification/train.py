import torch
import argparse
import time
#from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.datasets import PPI
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from gcn import GCN
from graphSAGE import GraphSAGE

# Arguments
parser = argparse.ArgumentParser(description='GNN arguments.')

parser.add_argument('--model_type', type=str,
                    help='Type of GNN model.')
parser.add_argument('--num_layers', type=int,
                    help='Number of graph conv layers')
parser.add_argument('--hidden_dim', type=int,
                    help='Training hidden size')
parser.add_argument('--aggregator', type=str,
                    help='Aggregator for GraphSAGE')
parser.add_argument('--epochs', type=int,
                    help='Number of training epochs')

parser.set_defaults(model_type='GCN',
                    num_layers=1,
                    hidden_dim=20,
                    aggregator='mean',
                    epochs=200)
args = parser.parse_args()

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
num_epochs = args.epochs
learning_rate = 0.001

# Inintialize tensorboard SummaryWriter
#writer = SummaryWriter('runs/ppi_mnc_3')

# Files for results
filenames = []
if (args.model_type == 'SAGE'):
    filenames.append('./results/SAGE/fold_results' +
                     'SAGE' + args.aggregator + str(args.num_layers) + '.txt')
    filenames.append('./results/SAGE/final_results' +
                     'SAGE' + args.aggregator + str(args.num_layers) + '.txt')
else:
    filenames.append('./results/GCN/fold_results' +
                     'GCN' + args.aggregator + str(args.num_layers) + '.txt')
    filenames.append('./results/GCN/final_results' +
                     'GCN' + args.aggregator + str(args.num_layers) + '.txt')
fold_res = open(filenames[0], 'w')
final_res = open(filenames[1], 'w')

# Start print
print('--------------------------------')
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    #print(len(train_ids), len(test_ids))
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    train_ids = train_ids.tolist()
    test_ids = test_ids.tolist()
    train_data = []
    test_data = []
    for id in train_ids:
        train_data.append(dataset[id])
    for id in test_ids:
        test_data.append(dataset[id])

    # Dataloader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

    # Initialize the model and
    if (args.model_type == 'SAGE'):
        model = GraphSAGE(train_dataset, args.num_layers,
                          args.hidden_dim, args.aggregator).to(device)
    elif (args.model_type == 'GCN'):
        model = GCN(train_dataset, args.num_layers,
                    args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run the training loop for defined number of epochs
    start_time = time.time()
    training_time = []
    for epoch in range(0, num_epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        total_loss = total_examples = 0.0

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
    end_time = time.time()
    print('Training time(in seconds): ', end_time - start_time)
    training_time.append(end_time - start_time)
    print('Loss: %f' % (total_loss / total_examples))
    #writer.add_scalar('training_loss', total_loss / total_examples, epoch)
    # writer.close()
    # sys.exit()

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
            #print("Output: ", out)
            preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    # print("Label: ", y)
    # print("Pred: ", pred)
    # Compute and Print metrics
    F1_Score = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    Precision = precision_score(
        y, pred, average='micro') if pred.sum() > 0 else 0
    Recall = recall_score(y, pred, average='micro') if pred.sum() > 0 else 0
    ROC_AUC_Score = roc_auc_score(y, pred)

    print('F1-score for fold %d: %f ' % (fold, F1_Score))
    print('Precision score for fold %d: %f' % (fold, Precision))
    print('Recall score for fold %d: %f ' % (fold, Recall))
    print('ROC-AUC score for fold %d: %f ' % (fold, ROC_AUC_Score))
    print('--------------------------------')
    results[fold] = (F1_Score, Precision, Recall, ROC_AUC_Score)
    # write results in file
    fold_res.writelines('F1-score for fold %d: %f \n' % (fold, F1_Score))
    fold_res.writelines('Precision score for fold %d: %f \n' %
                        (fold, Precision))
    fold_res.writelines('Recall score for fold %d: %f \n' % (fold, Recall))
    fold_res.writelines('ROC-AUC score for fold %d: %f \n' %
                        (fold, ROC_AUC_Score))
    fold_res.writelines('Training time(in seconds) %f: \n' %
                        (end_time-start_time))
    fold_res.writelines('\n----------------------------\n')

fold_res.close()

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
F1_sum = Recall_sum = Precision_sum = ROC_AUC_sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    F1_sum += value[0]
    Precision_sum += value[1]
    Recall_sum += value[2]
    ROC_AUC_sum += value[3]

training_time = np.array(training_time)
avg_training_time = np.mean(training_time)

#print(F1_sum, Precision_sum, Recall_sum)
print('Average F1-score: %f' % (F1_sum/len(results.items())))
print('Average precision-score: %f' % (Precision_sum/len(results.items())))
print('Average recall-score: %f' % (Recall_sum / len(results.items())))
print('Average ROC-AUC-score: %f' % (ROC_AUC_sum / len(results.items())))

# write results in file
final_res.writelines(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS\n')
final_res.writelines('Average F1-score: %f \n' % (F1_sum/len(results.items())))
final_res.writelines('Average precision-score: %f \n' %
                     (Precision_sum/len(results.items())))
final_res.writelines('Average recall-score: %f \n' %
                     (Recall_sum / len(results.items())))
final_res.writelines('Average ROC-AUC-score: %f \n' %
                     (ROC_AUC_sum / len(results.items())))
final_res.writelines('Average Training time(in seconds): %f \n' %
                     (avg_training_time))
final_res.close()

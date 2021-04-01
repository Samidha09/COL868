import os

model_type = ['SAGE', 'GCN']
aggr = 'max'
num_layers = [2, 3]
hidden_dim = [64, 128, 256, 512]
epochs = 300

#print(model_type, num_layers, hidden_dim, epochs, aggr)
# put below folder name intead of results/SAGE and GCN/SAGE in filename strings in train.py
os.system("mkdir dim_results")
os.system("mkdir ./dim_results/SAGE")
os.system("mkdir ./dim_results/GCN")

for model in model_type:
    for dim in hidden_dim:
        if model == 'SAGE':
            os.system("python3 train.py --model_type={} --num_layers={} --hidden_dim={} --epochs={} --aggregator={}".format(
                model, num_layers[1], dim, epochs, aggr))
        else:
            os.system("python3 train.py --model_type={} --num_layers={} --hidden_dim={} --epochs={}".format(
                      model, num_layers[0], dim, epochs))

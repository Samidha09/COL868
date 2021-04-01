import os

model_type = ['SAGE', 'GCN']
aggr = 'max'
num_layers = [2, 3]
hidden_dim = 256
epochs = [100, 200, 300, 400, 500, 600, 700]

#print(model_type, num_layers, hidden_dim, epochs, aggr)
# put below folder name intead of results/SAGE and GCN/SAGE in filename strings in train.py
os.system("mkdir plot_results")
os.system("mkdir ./plot_results/SAGE")
os.system("mkdir ./plot_results/GCN")

for model in model_type:
    for epoch in epochs:
        if model == 'SAGE':
            os.system("python3 train.py --model_type={} --num_layers={} --hidden_dim={} --epochs={} --aggregator={}".format(
                model, num_layers[1], hidden_dim, epoch, aggr))
        else:
            os.system("python3 train.py --model_type={} --num_layers={} --hidden_dim={} --epochs={}".format(
                      model, num_layers[0], hidden_dim, epoch))

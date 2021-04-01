import os

model_type = ['SAGE']  # , 'GCN']
aggr = ['mean']  # , 'add', 'max']
num_layers = [3]  # , 4, 5]
hidden_dim = 256
epochs = 10

#print(model_type, num_layers, hidden_dim, epochs, aggr)

for model in model_type:
    if model == 'SAGE':
        for ag in aggr:
            for layers in num_layers:
                os.system("python3 train.py --model_type={} --num_layers={} --hidden_dim={} --epochs={} --aggregator={}".format(
                    model, layers, hidden_dim, epochs, ag))
    else:
        for layers in num_layers:
            os.system("python3 train.py --model_type={} --num_layers={} --hidden_dim={} --epochs={}".format(
                model, layers, hidden_dim, epochs))

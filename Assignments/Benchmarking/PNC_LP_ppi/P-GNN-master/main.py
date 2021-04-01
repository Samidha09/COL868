from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter

from args import *
from model import *
from utils import *
from dataset import *
import time

if not os.path.isdir('results'):
    os.mkdir('results')
# args
args = make_args()
print(args)
np.random.seed(123)
np.random.seed()
writer_train = SummaryWriter(
    comment=args.task+'_'+args.model+'_'+args.comment+'_train')
writer_val = SummaryWriter(comment=args.task+'_' +
                           args.model+'_'+args.comment+'_val')
writer_test = SummaryWriter(
    comment=args.task+'_'+args.model+'_'+args.comment+'_test')


# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')


task = args.task
time_vs_metric = './results/' + args.task + \
    '_time_vs_metric_' + args.model + '_' + args.dataset + '.txt'
file_time_vs_metric = open(time_vs_metric, 'w')
if args.dataset == 'All':
    if task == 'link':
        datasets_name = ['grid', 'communities', 'ppi']
    else:
        datasets_name = ['communities', 'email', 'protein']
else:
    datasets_name = [args.dataset]
for dataset_name in datasets_name:
    # if dataset_name in ['communities','grid']:
    #     args.cache = False
    # else:
    #     args.epoch_num = 401
    #     args.cache = True
    results = []
    f1_results = []
    prec_results = []
    recall_results = []
    train_time = []

    for repeat in range(args.repeat_num):
        result_val = []
        result_test = []

        f1_results_val = []
        f1_results_test = []
        prec_results_val = []
        prec_results_test = []
        recall_results_val = []
        recall_results_test = []

        time1 = time.time()
        data_list = get_tg_dataset(
            args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
        time2 = time.time()
        print(dataset_name, 'load time',  time2-time1)

        num_features = data_list[0].x.shape[1]
        num_node_classes = None
        num_graph_classes = None
        if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
            num_node_classes = max([data.y.max().item()
                                    for data in data_list])+1
        if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
            num_graph_classes = max(
                [data.y_graph.numpy()[0] for data in data_list])+1
        print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features,
              'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
        nodes = [data.num_nodes for data in data_list]
        edges = [data.num_edges for data in data_list]
        print('Node: max{}, min{}, mean{}'.format(
            max(nodes), min(nodes), sum(nodes)/len(nodes)))
        print('Edge: max{}, min{}, mean{}'.format(
            max(edges), min(edges), sum(edges)/len(edges)))

        args.batch_size = min(args.batch_size, len(data_list))
        print('Anchor num {}, Batch size {}'.format(
            args.anchor_num, args.batch_size))

        # data
        for i, data in enumerate(data_list):
            preselect_anchor(data, layer_num=args.layer_num,
                             anchor_num=args.anchor_num, device='cpu')
            data = data.to(device)
            data_list[i] = data

        # model
        input_dim = num_features
        output_dim = args.output_dim
        model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                                     hidden_dim=args.hidden_dim, output_dim=output_dim,
                                     feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
        # loss
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=5e-4)
        if 'link' in args.task:
            loss_func = nn.BCEWithLogitsLoss()
            out_act = nn.Sigmoid()

        start_time = time.time()

        for epoch in range(args.epoch_num):
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
            model.train()
            optimizer.zero_grad()
            shuffle(data_list)
            effective_len = len(data_list)//args.batch_size*len(data_list)
            for id, data in enumerate(data_list[:effective_len]):
                if args.permute:
                    preselect_anchor(
                        data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                out = model(data)
                # get_link_mask(data,resplit=False)  # resample negative links
                edge_mask_train = np.concatenate(
                    (data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                nodes_first = torch.index_select(out, 0, torch.from_numpy(
                    edge_mask_train[0, :]).long().to(device))
                nodes_second = torch.index_select(out, 0, torch.from_numpy(
                    edge_mask_train[1, :]).long().to(device))
                pred = torch.sum(nodes_first * nodes_second, dim=-1)
                label_positive = torch.ones(
                    [data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros(
                    [data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                label = torch.cat(
                    (label_positive, label_negative)).to(device)
                loss = loss_func(pred, label)

                # update
                loss.backward()
                if id % args.batch_size == args.batch_size-1:
                    if args.batch_size > 1:
                        # if this is slow, no need to do this normalization
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad /= args.batch_size
                    optimizer.step()
                    optimizer.zero_grad()

            if epoch % args.epoch_log == 0:
                # evaluate
                model.eval()
                loss_train = 0
                loss_val = 0
                loss_test = 0
                correct_train = 0
                all_train = 0
                correct_val = 0
                all_val = 0
                correct_test = 0
                all_test = 0
                auc_train = 0
                auc_val = 0
                auc_test = 0
                f1_train = 0
                f1_val = 0
                f1_test = 0
                prec_train = 0
                prec_val = 0
                prec_test = 0
                recall_train = 0
                recall_val = 0
                recall_test = 0
                emb_norm_min = 0
                emb_norm_max = 0
                emb_norm_mean = 0
                for id, data in enumerate(data_list):
                    out = model(data)
                    emb_norm_min += torch.norm(out.data,
                                               dim=1).min().cpu().numpy()
                    emb_norm_max += torch.norm(out.data,
                                               dim=1).max().cpu().numpy()
                    emb_norm_mean += torch.norm(out.data,
                                                dim=1).mean().cpu().numpy()

                    # train
                    # get_link_mask(data, resplit=False)  # resample negative links
                    edge_mask_train = np.concatenate(
                        (data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(
                        edge_mask_train[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(
                        edge_mask_train[1, :]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones(
                        [data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros(
                        [data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                    label = torch.cat(
                        (label_positive, label_negative)).to(device)
                    loss_train += loss_func(pred, label).cpu().data.numpy()
                    auc_train += roc_auc_score(label.flatten().cpu(
                    ).numpy(), out_act(pred).flatten().data.cpu().numpy())
                    f1_train += f1_score(label.flatten().cpu().numpy(),
                                         (out_act(pred) >= 0.5).flatten().data.cpu().numpy())
                    prec_train += precision_score(label.flatten().cpu().numpy(
                    ), (out_act(pred) >= 0.5).flatten().data.cpu().numpy())
                    recall_train += recall_score(label.flatten().cpu().numpy(
                    ), (out_act(pred) >= 0.5).flatten().data.cpu().numpy())

                    end_time = time.time()
                    train_time.append(end_time - start_time)
                    # val
                    edge_mask_val = np.concatenate(
                        (data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                    nodes_first = torch.index_select(
                        out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                    nodes_second = torch.index_select(
                        out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones(
                        [data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros(
                        [data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                    label = torch.cat(
                        (label_positive, label_negative)).to(device)
                    loss_val += loss_func(pred, label).cpu().data.numpy()
                    auc_val += roc_auc_score(label.flatten().cpu().numpy(),
                                             out_act(pred).flatten().data.cpu().numpy())
                    f1_val += f1_score(label.flatten().cpu().numpy(),
                                       (out_act(pred) >= 0.5).flatten().data.cpu().numpy())
                    prec_val += precision_score(label.flatten().cpu().numpy(
                    ), (out_act(pred) >= 0.5).flatten().data.cpu().numpy())
                    recall_val += recall_score(label.flatten().cpu().numpy(
                    ), (out_act(pred) >= 0.5).flatten().data.cpu().numpy())

                    # test
                    edge_mask_test = np.concatenate(
                        (data.mask_link_positive_test, data.mask_link_negative_test), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(
                        edge_mask_test[0, :]).long().to(device))
                    nodes_second = torch.index_select(
                        out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones(
                        [data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros(
                        [data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                    label = torch.cat(
                        (label_positive, label_negative)).to(device)
                    loss_test += loss_func(pred, label).cpu().data.numpy()
                    #print("Labels: ", label.flatten().cpu().numpy())
                    # print("Pred: ", out_act(
                    #     pred).flatten().data.cpu().numpy())
                    auc_test += roc_auc_score(label.flatten().cpu().numpy(),
                                              out_act(pred).flatten().data.cpu().numpy())
                    f1_test += f1_score(label.flatten().cpu().numpy(),
                                        (out_act(pred) >= 0.5).flatten().data.cpu().numpy())
                    prec_test += precision_score(label.flatten().cpu().numpy(
                    ), (out_act(pred) >= 0.5).flatten().data.cpu().numpy())
                    recall_test += recall_score(label.flatten().cpu().numpy(
                    ), (out_act(pred) >= 0.5).flatten().data.cpu().numpy())

                loss_train /= id+1
                loss_val /= id+1
                loss_test /= id+1
                emb_norm_min /= id+1
                emb_norm_max /= id+1
                emb_norm_mean /= id+1
                auc_train /= id+1
                auc_val /= id+1
                auc_test /= id+1
                f1_train /= id+1
                f1_val /= id+1
                f1_test /= id + 1
                prec_train /= id+1
                prec_val /= id+1
                prec_test /= id+1
                recall_train /= id+1
                recall_val /= id+1
                recall_test /= id+1

                print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                      'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test))
                print('Train F1: {:.4f}'.format(f1_train),
                      'Val F1: {:.4f}'.format(f1_val), 'Test F1: {:.4f}'.format(f1_test))
                print('Train Precision: {:.4f}'.format(prec_train),
                      'Val Precision: {:.4f}'.format(prec_val), 'Test Precision: {:.4f}'.format(prec_test))
                print('Train Recall: {:.4f}'.format(recall_train),
                      'Val Recall: {:.4f}'.format(recall_val), 'Test Recall: {:.4f}'.format(recall_test))

                if(epoch % 100 == 0):
                    file_time_vs_metric.write('Epoch {}\n'.format(epoch))
                    file_time_vs_metric.write(
                        'Loss {:.4f}\n'.format(loss_train))
                    file_time_vs_metric.write(
                        'Train AUC: {:.4f}\t'.format(auc_train))
                    file_time_vs_metric.write(
                        'Val AUC: {:.4f}\t'.format(auc_val))
                    file_time_vs_metric.write(
                        'Test AUC: {:.4f}\n'.format(auc_test))
                    file_time_vs_metric.write(
                        'Train F1: {:.4f}\t'.format(f1_train))
                    file_time_vs_metric.write(
                        'Val F1: {:.4f}\t'.format(f1_val))
                    file_time_vs_metric.write(
                        'Test F1: {:.4f}\n'.format(f1_test))
                    file_time_vs_metric.write(
                        'Train Precision: {:.4f}\t'.format(prec_train))
                    file_time_vs_metric.write(
                        'Val Precision: {:.4f}\t'.format(prec_val))
                    file_time_vs_metric.write(
                        'Test Precision: {:.4f}\n'.format(prec_test))
                    file_time_vs_metric.write(
                        'Train Recall: {:.4f}\t'.format(recall_train))
                    file_time_vs_metric.write(
                        'Val Recall: {:.4f}\t'.format(recall_val))
                    file_time_vs_metric.write(
                        'Test Recall: {:.4f}\n'.format(recall_test))
                    file_time_vs_metric.write(
                        '\n ----------------------------- \n')

                writer_train.add_scalar(
                    'repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
                writer_train.add_scalar(
                    'repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
                writer_val.add_scalar(
                    'repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
                writer_train.add_scalar(
                    'repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)
                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/auc_'+dataset_name, auc_test, epoch)
                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/loss_'+dataset_name, loss_test, epoch)
                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/emb_min_'+dataset_name, emb_norm_min, epoch)
                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/emb_max_'+dataset_name, emb_norm_max, epoch)
                writer_test.add_scalar(
                    'repeat_' + str(repeat) + '/emb_mean_'+dataset_name, emb_norm_mean, epoch)
                result_val.append(auc_val)
                result_test.append(auc_test)
                f1_results_val.append(f1_val)
                f1_results_test.append(f1_test)
                prec_results_val.append(prec_val)
                prec_results_test.append(prec_test)
                recall_results_val.append(recall_val)
                recall_results_test.append(recall_test)

        result_val = np.array(result_val)
        result_test = np.array(result_test)
        results.append(result_test[np.argmax(result_val)])

        f1_results_val = np.array(f1_results_val)
        f1_results_test = np.array(f1_results_test)
        f1_results.append(f1_results_test[np.argmax(f1_results_val)])

        prec_results_val = np.array(prec_results_val)
        prec_results_test = np.array(prec_results_test)
        prec_results.append(prec_results_test[np.argmax(prec_results_val)])

        recall_results_val = np.array(recall_results_val)
        recall_results_test = np.array(recall_results_test)
        recall_results.append(
            recall_results_test[np.argmax(recall_results_val)])

    results = np.array(results)
    results_mean = np.mean(results).round(6)
    results_std = np.std(results).round(6)

    f1_results = np.array(f1_results)
    f1_results_mean = np.mean(f1_results).round(6)
    f1_results_std = np.std(f1_results).round(6)

    prec_results = np.array(prec_results)
    prec_results_mean = np.mean(prec_results).round(6)
    prec_results_std = np.std(prec_results).round(6)

    recall_results = np.array(recall_results)
    recall_results_mean = np.mean(recall_results).round(6)
    recall_results_std = np.std(recall_results).round(6)

    average_train_time_per_fold = np.mean(np.array(train_time))
    file_time_vs_metric.close()
    print('-----------------Final-------------------')
    print(results_mean, results_std)
    print("F1 score: ", f1_results_mean, f1_results_std)
    print("Precision: ", prec_results_mean, prec_results_std)
    print("Recall: ", recall_results_mean, recall_results_std)
    with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task, args.model, dataset_name, args.layer_num, args.approximate), 'w') as f:
        f.write('{}, {}\n'.format(results_mean, results_std))

    filename = args.task + '_results_' + args.model + '.txt'
    out = open(filename, 'w')
    out.write('Task: {} - Results\n'.format(args.task))
    out.write('F1-score: {} +/- {} \n'.format(f1_results_mean,
                                              f1_results_std))
    out.write('Precision: {} +/- {} \n'.format(prec_results_mean,
                                               prec_results_std))
    out.write('Recall: {} +/- {} \n'.format(recall_results_mean,
                                            recall_results_std))
    out.write('ROC-AUC: {} +/- {} \n'.format(results_mean,
                                             results_std))
    out.write('Training time per fold : {}'.format(
        average_train_time_per_fold))
    out.close()

# export scalar data to JSON for external processing
writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()

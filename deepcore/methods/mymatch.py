from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import pickle
import collections
from ..methods.methods_utils.opt import *

class mymatch(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, repeat=5,
                 specific_model=None, dst_val=None,balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat
        self.dst_val = dst_val

        self.balance = balance
        # print("| 2Fraction[%.1f] Epochs[%3d]" % (fraction, epochs))

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    """
    处理数据
    """
    def _load_data(self, save_features):
        data = save_features
        labels = [np.full(shape=len(data[key]), fill_value=key) for key in data]
        indexs = [features[0] for key in data for features in data[key]]
        data_arr = [features[1] for key in data for features in data[key]]
        dataset = dict()
        dataset["data"] = torch.FloatTensor(np.stack(data_arr, axis=0))
        dataset["index"] = torch.LongTensor(indexs)
        dataset["labels"] = torch.LongTensor(np.concatenate(labels))
        return dataset
    """
    将数据传入模型开始处理数据
    """
    def define_data(self, val): 
        self.model.embedding_recorder.record_embedding = True
        self.model.eval()
        cfg = {}
        if val:
            val_batch_loader = torch.utils.data.DataLoader(self.dst_val, batch_size=self.args.selection_batch,
                                                           num_workers=self.args.workers)
            train_batch_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.selection_batch,
                                                             num_workers=self.args.workers)
            with torch.no_grad():
                output_dict = collections.defaultdict(list)
                j = 0
                for i, (inputs, labels) in enumerate(val_batch_loader):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs, _ = self.model(inputs)
                    outputs = outputs.cpu().data.numpy()
                    for out, label in zip(outputs, labels):
                        indx_feature = [j, out]
                        j += 1
                        output_dict[label.item()].append(indx_feature)
                k = 0
                for i, (inputs, labels) in enumerate(train_batch_loader):
                    inputs = inputs.to(self.args.device)
                    labels = labels.to(self.args.device)
                    _ = self.model(inputs)
                    outputs = self.model.embedding_recorder.embedding
                    outputs = outputs.cpu().data.numpy()
                    for out, label in zip(outputs, labels):
                        indx_feature = [j + k, out]
                        k += 1
                        output_dict[label.item()].append(indx_feature)
        else:
            train_batch_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.selection_batch,
                                                             num_workers=self.args.workers)
            save_features = []
            save_labels = []
            with torch.no_grad():
                output_dict = collections.defaultdict(list)
                j = 0
                # reload train set
                for k in range(2):
                    for i, (inputs, labels) in enumerate(train_batch_loader):
                        inputs = inputs.to(self.args.device)
                        labels = labels.to(self.args.device)
                        _ = self.model(inputs)
                        if self.args.approximate: # 取最后一层特征
                            outputs = _
                            outputs = outputs.cpu().data.numpy()
                            for out, label in zip(outputs, labels):
                                indx_feature = [j, out]
                                j += 1
                                output_dict[label.item()].append(indx_feature)
                        else:
                            outputs = self.model.embedding_recorder.embedding
                            outputs = outputs.cpu().data.numpy()
                            for out, label in zip(outputs, labels):
                                indx_feature = [j, out]
                                if k==0:
                                    save_features.append(out)
                                    # print("labels___:", label.item())
                                    save_labels.append(int(label.item()))
                                j += 1
                                output_dict[label.item()].append(indx_feature)
        # out_features = "/home/xww/tsne-pytorch/features.csv"
        # print("____________________", len(save_features), len(save_labels))
        # np.savetxt(out_features,np.array(save_features))
        # out_labels = "/home/xww/tsne-pytorch/labels.csv"
        # np.savetxt(out_labels, save_labels)
        # print("=====>  save end")
        self.model.train()
        self.model.embedding_recorder.record_embedding = False

        dataset = self._load_data(output_dict)
        print("**************dataset description*************")
        print(" train set :", len(self.dst_train.targets))
        print(" labels : ", len(torch.unique(dataset["labels"])), dataset["labels"].shape[0])
        print("***************description********************")
        # print("_______________0", torch.unique(dataset["labels"]), dataset["labels"].shape[0], len(torch.unique(dataset["labels"])), int(len(self.dst_train.targets)))
        _min_examples = int(dataset["labels"].shape[0]) / len(torch.unique(dataset["labels"]))
        if val:
            cfg["shot"] = int(len(self.dst_val) / len(torch.unique(dataset["labels"])))
        else:
            cfg["shot"] = int(len(self.dst_train.targets) / len(torch.unique(dataset["labels"])))
        cfg["ways"] = len(torch.unique(dataset["labels"]))
        cfg["queries"] = int(len(self.dst_train.targets) / len(torch.unique(dataset["labels"])))
        print("=> _min_examples : ", _min_examples)
        print("=> n_ways : ", cfg["ways"])
        print("=> shot : ", cfg["shot"])
        print("=> queries : ", cfg["queries"])
        data = torch.zeros((0, int(_min_examples), dataset["data"].shape[1]))
        labels = dataset["labels"].clone()
        index_all = torch.zeros(0, int(_min_examples))
        classes_name = []
        while labels.shape[0] > 0:
            indices = torch.where(dataset["labels"] == labels[0])[0]
            data = torch.cat([data, dataset["data"][indices, :]
            [:int(_min_examples)].view(1, int(_min_examples), -1)], dim=0)
            index_all = torch.cat([index_all, dataset["index"][indices].view(1, int(_min_examples))], dim=0)
            classes_name.append(int(dataset["labels"][int(indices[0])]))
            # print(" the i-class index : ", index_all.shape, "classes_name :", dataset["labels"][int(indices[0])])
            indices = torch.where(labels != labels[0])[0]
            labels = labels[indices]
        dataset_setting = None
        dataset_setting = torch.zeros((cfg['ways'], cfg['shot'] + cfg['queries'], data.shape[2]))
        for i in range(cfg["ways"]):
            dataset_setting[i] = data[classes_name[i], :, :][:cfg['shot'] + cfg['queries']]
            # print("=======>test===:", dataset_setting[i])
        return dataset_setting, index_all.numpy().astype(int), cfg
    def proccess_data(self):
        ndatas, all_index, cfg = self.define_data(val=False)  # 10, 10000, 512
        # print("======>111111111+++", ndatas[1])
        ndatas = ndatas.unsqueeze(0)
        print("=> redefine data shape : ", ndatas.shape)
        n_runs = 1
        n_lsamples = cfg["ways"] * cfg["shot"]
        n_usamples = cfg["ways"] * cfg["queries"]
        n_samples = n_lsamples + n_usamples
        ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1) #[3, 10000, 640]
        labels = torch.arange(cfg["ways"]).view(1,1,cfg["ways"]).expand(n_runs,cfg["shot"]+cfg["queries"],cfg["ways"]).clone().view(n_runs, n_samples)
        
        # Power transform
        beta = 0.5
        ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta)
        # ndatas = QRreduction(ndatas)
        n_nfeat = ndatas.size(2)
        ndatas = centerDatas(ndatas, n_lsamples)  #中心化
    
        print("=> size of the datas...", ndatas.size())
       
        # switch to cuda
        ndatas = ndatas.cuda()
        print("========> ndatas:", ndatas.shape)  #（1，200000，512）
        labels = labels.cuda()
        
        # # ***************Split Blocks MAP************
        # num, test_nums, feature_size = ndatas.size()
        # n = 10
        # return_result = torch.zeros((num,test_nums, feature_size)).cuda()
        # temp_ndatas = torch.zeros((num,int(test_nums/4), feature_size)).cuda()
        # for i in range(n):
        #     temp_ndatas = ndatas[:,i*int(test_nums/10):(i+1)*int(test_nums/10),:]
        #     print("show shape____", temp_ndatas.shape)
        #     lam = 10
        #     model = GaussianModel(cfg["ways"], lam, temp_ndatas, n_runs, cfg["shot"], cfg["queries"], n_nfeat, all_index, labels)
        #     model.initFromLabelledDatas()
        #     _, result = model.getProbas()
        #     print("sssssssss = ", type(result))
        # # ********************end*********************
        #MAP
        lam = 10
        model = GaussianModel(cfg["ways"], lam, ndatas, n_runs, cfg["shot"], cfg["queries"], n_nfeat, all_index, labels)
        model.initFromLabelledDatas()
        _, result = model.getProbas()
        return result, cfg
        # alpha = 0.2
        # optim = MAP(labels, n_runs, n_ways, n_shot, n_queries, alpha)

    def finish_run(self):  # 训练一定的epoch之后开始数据选择
        self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()
        ndatas, all_index, _ = self.define_data(val=False)
        result, cfg = self.proccess_data()
        return result, cfg, all_index
    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.run()
        result, cfg, all_index = self.finish_run()
        n_lsamples = cfg["ways"] * cfg["shot"]
        n_usamples = cfg["ways"] * cfg["queries"]
        n_samples = n_lsamples + n_usamples
        print("=> the result = ", result, len(result))
        self.train_indx = np.arange(self.n_train)
        if not self.balance:
            top_examples = np.array([], dtype=np.int64)
            print("======*******======>>>:", torch.argsort(result).cpu().numpy())
            # index_save = self.train_indx[(torch.argsort(result)).cpu().numpy()][:self.coreset_size]
            index_save = self.train_indx[(torch.argsort(result)).cpu().numpy()][-self.coreset_size:]
            print("======&&&&&&&======>>>:", index_save)
            top_examples = np.append(top_examples, index_save)
            # with open("./select_index.txt", "w") as f1:
            #     f1.write(str(tuple(index_save))+str("\n"))
            #     f1.close()
            # with open("./select_context.txt", "w") as f:
            #     index_save1, _ = torch.sort(result)
            #     f.write(str(tuple(result.cpu().numpy()))+str("\n"))
            #     f.close()
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c] 
                budget = round(self.fraction * len(c_indx))
                print("=> budget : ", budget)
                index_save = (torch.argsort(result[all_index[c][cfg["shot"]:]-int(n_lsamples)]).cpu().numpy())[:budget]
                top_examples = np.append(top_examples, index_save)
                # c_indx = self.train_indx[self.dst_train.targets == c]
                # budget = round(self.fraction * len(c_indx))
                # print("___________", c_indx, budget, np.argsort(result[[]-int(n_lsamples)]))
                # top_examples = np.append(top_examples, c_indx[np.argsort(result[all_index[c][cfg["shot"]:]-int(n_lsamples)])][:budget])
            with open("./select_context_balance.txt", "w") as f:
                for c in range(self.num_classes):
                    c_index = self.train_indx[self.dst_train.targets == c] 
                    budget = round(self.fraction * len(c_index))
                    # index_save = self.train_indx[(torch.argsort(result)).cpu().numpy()]
    
                    index_save = torch.argsort(result[c_index]).cpu().numpy()
                    print("==> show result = ", index_save)
                    f.write(str(tuple(index_save))+str("\n"))
                f.close()
        return {"indices": top_examples, "scores": result}

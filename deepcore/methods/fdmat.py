from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import collections
from torch import tensor, long
def compute_optimal_transport(M, r, c, epsilon=1e-6, lam=4):    
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape  
        P = torch.exp(- lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)                        
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, P*M 

class fdmat(EarlyTrain):
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

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module


    # load data
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

    # Define data format
    def define_data(self, val): 
        self.model.embedding_recorder.record_embedding = True
        self.model.eval()
        train_batch_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.selection_batch,
                                                             num_workers=self.args.workers)
        save_features = []
        save_labels = []
        with torch.no_grad():
            output_dict = collections.defaultdict(list)
            j = 0
            # loading train samples
            for i, (inputs, labels) in enumerate(train_batch_loader):
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = self.model(inputs)
                # getting the feature embedding of the last layer 
                if self.args.approximate:
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
                        save_features.append(out)
                        save_labels.append(int(label.item()))
                        j += 1
                        output_dict[label.item()].append(indx_feature)
        self.model.train()
        self.model.embedding_recorder.record_embedding = False

        # getting data index labels
        dataset = self._load_data(output_dict)
        labels = dataset["labels"].clone()
        classes_name = []

        each_class_examples = []
        save_each_data = []
        save_each_data_class = []
        while labels.shape[0] > 0:
            indices = torch.where(dataset["labels"] == labels[0])[0]
            each_class_examples.append(len(indices))
            save_each_data.append(dataset["data"][indices,:])
            save_each_data_class.append(indices)
            classes_name.append(int(dataset["labels"][int(indices[0])]))
            indices = torch.where(labels != labels[0])[0]
            labels = labels[indices]
        return save_each_data, save_each_data_class
    def proccess_data(self): # 
        new_datas, all_data_index = self.define_data(val=False)  
        # Power transform
        for i in range(len(new_datas)):
            beta = 0.5
            new_datas[i] = torch.pow(new_datas[i]+1e-6, beta)
        # CenterDatas
        cat_datas = torch.cat(new_datas, dim=0)
        new_datas_mean = cat_datas.mean(0, keepdim=True)
        new_datas_norm = torch.norm(cat_datas,2, 1)[:, None]
        num = 0
        new_datas_indices = [0]
        class_indices = []
        for i in range(len(new_datas)):
            num += len(new_datas[i])
            new_datas_indices.append(num)
            class_indices.append(len(new_datas[i]))
        data_mus = []
        for i in range(len(new_datas)):
            new_datas[i] = ((cat_datas[new_datas_indices[i]:new_datas_indices[i+1],:] - new_datas_mean)/new_datas_norm[new_datas_indices[i]:new_datas_indices[i+1],:]) #(10000, 1)
            data_mus.append(new_datas[i].mean(0).unsqueeze(0))
        data_mus = torch.cat(data_mus, dim=0)  
        datas = torch.cat(new_datas, dim=0).unsqueeze(0)
        mus = data_mus.unsqueeze(0) 

        # define cost matrix
        all_samples = datas.size()[1]
        num_class = data_mus.size()[0]
        dist = torch.zeros((1,all_samples,num_class)).cuda()
        for i in range(num_class):
            dist[:,i*int(all_samples/num_class):(i+1)*int(all_samples/num_class),:] = (datas[:,i*int(all_samples/num_class):(i+1)*int(all_samples/num_class),].unsqueeze(2)-mus.unsqueeze(1)).norm(dim=3).pow(2)
        n_runs = 1
        n_usamples = int(all_samples)
            
        # select the coreset by uisng optimal transport
        r = torch.ones(n_runs, n_usamples) 
        c = torch.ones(n_runs, num_class) * torch.tensor(class_indices)
        p_xj_test, all_sum = compute_optimal_transport(dist[:, :], r, c, epsilon=1e-6, lam=self.args.lmda)
        result = all_sum.norm(dim=2).pow(2)[0]
        return result

    def finish_run(self):  # get the pre-trained model by training a few epoches and select coreset
        self.model.embedding_recorder.record_embedding = True  # recording embedding vector
        self.model.eval()
        b, all_index= self.define_data(val=False)
        result = self.proccess_data()
        return result, all_index
    def select(self, **kwargs):
        self.run()
        result,  all_index = self.finish_run()
        self.train_indx = np.arange(self.n_train)
        if not self.balance:
            top_examples = np.array([], dtype=np.int64)
            index_save = self.train_indx[(torch.argsort(result)).cpu().numpy()][:self.coreset_size]
            top_examples = np.append(top_examples, index_save)

        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = torch.tensor(all_index[c])
                budget = round(self.fraction * len(c_indx))
                result = tensor(result, dtype=long)
                index_save = (torch.argsort(result[c_indx]).cpu().numpy())[:budget]
                index_save = c_indx[index_save]
                top_examples = np.append(top_examples, index_save)
        return {"indices": top_examples, "scores": result}

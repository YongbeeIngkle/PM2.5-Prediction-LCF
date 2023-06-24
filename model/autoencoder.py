import os
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader

class AutoencoderCnn(nn.Module):
    def __init__(self, encode_dim, **input_shapes):
        super(AutoencoderCnn, self).__init__()
        self.input_dim = input_shapes['input_dim']
        self.input_len = input_shapes['input_len']
        self.encode_dim = encode_dim
        self._define_encoder()
        self._define_decoder()
        self._define_pm_estimator()

    def _define_encoder(self):
        self.enocde_layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding="same")
        if self.encode_dim == 0:
            self.encode_layer4 = nn.Linear(4*self.input_len, 1)
        else:
            self.encode_layer4 = nn.Linear(4*self.input_len, self.encode_dim*self.input_len)

    def _define_decoder(self):
        self.deocde_layer1 = nn.Conv1d(in_channels=self.encode_dim, out_channels=4, kernel_size=3, padding="same")
        self.decode_layer2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding="same")
        self.decode_layer3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.decode_layer4 = nn.Conv1d(in_channels=16, out_channels=self.input_dim, kernel_size=1)

    def _define_pm_estimator(self):
        self.estimator_layer = nn.Linear(self.encode_dim*self.input_len, 1)

    def encode(self, x):
        x = torch.relu(self.enocde_layer1(x))
        x = torch.relu(self.encode_layer2(x))
        x = torch.relu(self.encode_layer3(x))
        x = x.view(-1, 4*self.input_len)
        x = self.encode_layer4(x)
        return x
    
    def decode(self, x):
        x = x.view(-1, self.encode_dim, self.input_len)
        x = torch.relu(self.deocde_layer1(x))
        x = torch.relu(self.decode_layer2(x))
        x = torch.relu(self.decode_layer3(x))
        x = self.decode_layer4(x)
        return x
    
    def estimate(self, x):
        x = self.encode(x)
        x = self.estimator_layer(x)
        x = x.flatten()
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoencoderFeature(nn.Module):
    def __init__(self, **input_shapes):
        super(AutoencoderFeature, self).__init__()
        self.input_dim = input_shapes['input_dim']
        self.input_len = input_shapes['input_len']
        self._define_encoder()
        self._define_decoder()
        self._define_pm_estimator()

    def _define_encoder(self):
        self.enocde_layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=1)
        self.encode_layer2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.encode_layer3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding="same")
        self.encode_layer4 = nn.Linear(4*self.input_len, 1)

    def _define_decoder(self):
        self.deocde_layer1 = nn.Linear(1, self.input_len)
        self.decode_layer2 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding="same")
        self.decode_layer3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.decode_layer4 = nn.Conv1d(in_channels=16, out_channels=self.input_dim, kernel_size=1)

    def _define_pm_estimator(self):
        self.estimator_layer = nn.Linear(1, 1)

    def encode(self, x):
        x = torch.relu(self.enocde_layer1(x))
        x = torch.relu(self.encode_layer2(x))
        x = torch.relu(self.encode_layer3(x))
        x = x.view(-1, 4*self.input_len)
        x = self.encode_layer4(x)
        return x
    
    def decode(self, x):
        x = torch.relu(self.deocde_layer1(x))
        x = x.view(-1, 1, self.input_len)
        x = torch.relu(self.decode_layer2(x))
        x = torch.relu(self.decode_layer3(x))
        x = self.decode_layer4(x)
        return x
    
    def estimate(self, x):
        x = self.encode(x)
        x = self.estimator_layer(x)
        x = x.flatten()
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class AutoencodeModel:
    def __init__(self, model):
        self.model = model
        self.loss_function = MSELoss()

    def fit(self, train_loader: DataLoader, epoch: int):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=0.001)
        
        for e in range(epoch):
            total_data_num = 0
            total_recon_loss, total_pm_loss = 0, 0
            for train_x, train_pm in train_loader:
                train_x = train_x.float()
                train_pm = train_pm.float()
                recon = self.model(train_x).float()
                pred = self.model.estimate(train_x).float()
                recon_loss = self.loss_function(recon, train_x) 
                pm_loss = self.loss_function(pred, train_pm)
                total_recon_loss += (recon_loss*train_x.size(0)).item()
                total_pm_loss += (pm_loss*train_x.size(0)).item()
                total_data_num += train_x.size(0)

                optimizer.zero_grad()
                recon_loss.backward()
                pm_loss.backward()
                optimizer.step()
            mean_recon_loss = total_recon_loss/total_data_num
            mean_pm_loss = total_pm_loss/total_data_num
            print(f"Epoch{e+1} - Mean Reconstruction Loss: {mean_recon_loss}, Mean PM2.5 Loss: {mean_pm_loss}")

    def encode(self, encode_loader: DataLoader):
        all_encode_vals = []
        for encode_x, _ in encode_loader:
            encode_x = encode_x.float()
            encode_val = self.model.encode(encode_x).float()
            all_encode_vals.append(encode_val.detach().numpy())
        all_encode_vals = np.vstack(all_encode_vals)
        return all_encode_vals
    
    def freeze(self):
        layers_to_freeze = [
            'deocde_layer1', 'decode_layer2', 'decode_layer3', 'decode_layer4', 'estimator_layer'
        ]
        for name, module in self.model.named_modules():
            if name in layers_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

class TrainWholeModel:
    def __init__(self, model_name: str, encode_dim: int, input_shape: dict, model_dir: str):
        self.model_name = model_name
        self.model_dir = model_dir
        self.encode_dim = encode_dim
        self.input_shape = input_shape

    def define_model(self, input_shape):
        if self.model_name == "LSV":
            model = AutoencoderCnn(self.encode_dim, input_dim=input_shape[0], input_len=input_shape[1])
        elif self.model_name == "LF":
            model = AutoencoderFeature(input_dim=input_shape[0], input_len=input_shape[1])
        return AutoencodeModel(model)

    def train(self, train_dataset: dict, train_num: int, epoch: int):
        self.all_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Autoencoder Train")
            split_model_dir = f"{self.model_dir}train{train_num}_split{cluster_id}"
            if os.path.exists(split_model_dir):
                model = torch.load(split_model_dir)
            else:
                data_loader = train_dataset[cluster_id]
                model = self.define_model(self.input_shape[cluster_id])
                model.fit(data_loader, epoch)
                torch.save(model, split_model_dir)
            self.all_models[cluster_id] = model
    
    def test_encode(self, test_dataset: dict):
        all_encode_vals = {}
        for cluster_id in test_dataset.keys():
            data_loader = test_dataset[cluster_id]
            encode_val = self.all_models[cluster_id].encode(data_loader)
            all_encode_vals[cluster_id] = encode_val
        return all_encode_vals
    
    def compose_feature(self, test_dataset: dict):
        all_encode_vals = {}
        for cluster_id in test_dataset.keys():
            data_loader = test_dataset[cluster_id]
            encode_val = self.all_models[cluster_id].encode(data_loader)
            all_encode_vals[cluster_id] = encode_val
        return all_encode_vals

class EncodeTuningModel:
    def __init__(self, model_name: str, encode_dim: int, input_shape: dict):
        self.model_name = model_name
        self.encode_dim = encode_dim
        self.input_shape = input_shape

    def define_model(self, input_shape):
        if self.model_name == 'CNN':
            model = AutoencoderCnn(self.encode_dim, input_dim=input_shape[0], input_len=input_shape[1])
        return AutoencodeModel(model)

    def source_train(self, train_dataset: dict, epoch: int):
        self.source_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Train")
            data_loader = train_dataset[cluster_id]
            model = self.define_model(self.input_shape[cluster_id])
            model.fit(data_loader, epoch)
            self.source_models[cluster_id] = model

    def fine_tune_train(self, train_dataset: dict, epoch: int):
        self.target_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Fine-tune Train")
            data_loader = train_dataset[cluster_id]
            state_dict = self.source_models[cluster_id].model.state_dict()
            model = self.define_model(self.input_shape[cluster_id])
            model.model.load_state_dict(state_dict)
            model.freeze()
            model.fit(data_loader, epoch)
            self.target_models[cluster_id] = model

    def train_encode(self, source_dataset: dict, train_target_dataset: dict):
        all_encode_vals = {}
        for cluster_id in source_dataset.keys():
            source_loader = source_dataset[cluster_id]
            train_loader = train_target_dataset[cluster_id]
            source_encode = self.source_models[cluster_id].encode(source_loader)
            train_encode = self.target_models[cluster_id].encode(train_loader)
            all_encode_vals[cluster_id] = np.vstack([source_encode, train_encode])
        return all_encode_vals
    
    def test_encode(self, test_dataset: dict):
        all_encode_vals = {}
        for cluster_id in test_dataset.keys():
            data_loader = test_dataset[cluster_id]
            encode_val = self.target_models[cluster_id].encode(data_loader)
            all_encode_vals[cluster_id] = encode_val
        return all_encode_vals

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

from torch_utils.trainer import TorchTrainer
from ts_models.decoders import DecoderCell, AttentionDecoderCell
from ts_models.encoder_decoder import EncoderDecoderWrapper
from ts_models.encoders import RNNEncoder

warnings.filterwarnings('ignore')
tqdm.pandas()

torch.manual_seed(420)
np.random.seed(420)


def smape_loss(y_pred, y_true):
    denominator = (y_true + y_pred) / 200.0
    diff = torch.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return torch.mean(diff)


def smape_exp_loss(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(y_true)
    denominator = (y_true + y_pred) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


def differentiable_smape_loss(y_pred, y_true):
    epsilon = 0.1
    summ = torch.max(torch.abs(y_true) + torch.abs(y_pred) + epsilon, torch.tensor(0.5 + epsilon, device='cuda'))
    smape = torch.abs(y_pred - y_true) / summ
    return torch.mean(smape)


class StoreItemDataset(Dataset):
    def __init__(self, cat_columns=[], num_columns=[], embed_vector_size=None, decoder_input=True,
                 ohe_cat_columns=False):
        super().__init__()
        self.sequence_data = None
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.cat_classes = {}
        self.cat_embed_shape = []
        self.cat_embed_vector_size = embed_vector_size if embed_vector_size is not None else {}
        self.pass_decoder_input = decoder_input
        self.ohe_cat_columns = ohe_cat_columns
        self.cat_columns_to_decoder = False

    def get_embedding_shape(self):
        return self.cat_embed_shape

    def load_sequence_data(self, processed_data):
        self.sequence_data = processed_data

    def process_cat_columns(self, column_map=None):
        column_map = column_map if column_map is not None else {}
        for col in self.cat_columns:
            self.sequence_data[col] = self.sequence_data[col].astype('category')
            if col in column_map:
                self.sequence_data[col] = self.sequence_data[col].cat.set_categories(column_map[col]).fillna('#NA#')
            else:
                self.sequence_data[col].cat.add_categories('#NA#', inplace=True)
            self.cat_embed_shape.append(
                (len(self.sequence_data[col].cat.categories), self.cat_embed_vector_size.get(col, 50)))

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        row = self.sequence_data.iloc[[idx]]
        x_inputs = [torch.tensor(row['x_sequence'].values[0], dtype=torch.float32)]
        y = torch.tensor(row['y_sequence'].values[0], dtype=torch.float32)
        if self.pass_decoder_input:
            decoder_input = torch.tensor(row['y_sequence'].values[0][:, 1:], dtype=torch.float32)
        if len(self.num_columns) > 0:
            for col in self.num_columns:
                num_tensor = torch.tensor([row[col].values[0]], dtype=torch.float32)
                num_tensor_repeat = num_tensor.repeat(x_inputs[0].size(0))
                num_tensor_repeat_add_dim = num_tensor_repeat.unsqueeze(1)
                x_inputs[0] = torch.cat((x_inputs[0], num_tensor_repeat_add_dim), axis=1)
                decoder_input = torch.cat((decoder_input, num_tensor.repeat(decoder_input.size(0)).unsqueeze(1)),
                                          axis=1)
        if len(self.cat_columns) > 0:
            if self.ohe_cat_columns:
                for ci, (num_classes, _) in enumerate(self.cat_embed_shape):
                    col_tensor = torch.zeros(num_classes, dtype=torch.float32)
                    col_tensor[row[self.cat_columns[ci]].cat.codes.values[0]] = 1.0
                    col_tensor_x = col_tensor.repeat(x_inputs[0].size(0), 1)
                    x_inputs[0] = torch.cat((x_inputs[0], col_tensor_x), axis=1)
                    if self.pass_decoder_input and self.cat_columns_to_decoder:
                        col_tensor_y = col_tensor.repeat(decoder_input.size(0), 1)
                        decoder_input = torch.cat((decoder_input, col_tensor_y), axis=1)
            else:
                cat_tensor = torch.tensor(
                    [row[col].cat.codes.values[0] for col in self.cat_columns],
                    dtype=torch.long
                )
                x_inputs.append(cat_tensor)
        if self.pass_decoder_input:
            x_inputs.append(decoder_input)
            y = torch.tensor(row['y_sequence'].values[0][:, 0], dtype=torch.float32)
        if len(x_inputs) > 1:
            return tuple(x_inputs), y
        return x_inputs[0], y


# mode = 'valid'
mode = 'test'
sequence_data = pd.read_pickle('./sequence_data/sequence_data_stdscaler_test.pkl')
lag_null_filter = sequence_data['y_sequence'].apply(lambda val: np.isnan(val[:, -1].reshape(-1)).sum() == 0)
test_sequence_data = sequence_data[sequence_data['date'] == '2018-01-01']
# data after 10th month will have prediction data in y_sequence
if mode == 'test':
    train_sequence_data = sequence_data[
        (sequence_data['date'] <= '2017-10-01') & ((sequence_data['date'] >= '2014-01-02'))]
    valid_sequence_data = pd.DataFrame()
else:
    train_sequence_data = sequence_data[
        (sequence_data['date'] <= '2016-10-01') & (sequence_data['date'] >= '2014-01-02')]
    valid_sequence_data = sequence_data[
        (sequence_data['date'] > '2016-10-01') & (sequence_data['date'] <= '2017-01-01')]

train_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'],
                                 embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
valid_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'],
                                 embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)
test_dataset = StoreItemDataset(cat_columns=['store', 'item'], num_columns=['yearly_corr'],
                                embed_vector_size={'store': 4, 'item': 4}, ohe_cat_columns=True)

train_dataset.load_sequence_data(train_sequence_data)
valid_dataset.load_sequence_data(valid_sequence_data)
test_dataset.load_sequence_data(test_sequence_data)

cat_map = train_dataset.process_cat_columns()

if mode == 'valid':
    valid_dataset.process_cat_columns(cat_map)
test_dataset.process_cat_columns(cat_map)

batch_size = 256

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
(X_con, X_dec), y = next(iter(train_dataloader))

device = 'cuda'

encoder = RNNEncoder(
    input_feature_len=71,
    rnn_num_layers=1,
    hidden_size=100,
    sequence_len=180,
    bidirectional=False,
    device=device,
    rnn_dropout=0.2
)

# decoder_cell = DecoderCell(
#     input_feature_len=10,
#     hidden_size=100,
# )

decoder_cell =AttentionDecoderCell(input_feature_len=10,
    hidden_size=100,sequence_len=180)

# loss_function = differentiable_smape_loss
# loss_function = differentiable_smape_loss
loss_function = nn.MSELoss()
# loss_function = nn.SmoothL1Loss()
# encoder_optimizer = COCOBBackprop(encoder.parameters(), weight_decay=0)
# decoder_optimizer = COCOBBackprop(decoder_cell.parameters(), weight_decay=0)
# encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-3, weight_decay=1e-)
# decoder_optimizer = torch.optim.AdamW(decoder_cell.parameters(), lr=2e-3, weight_decay=1e-1)


encoder = encoder.to(device)
decoder_cell = decoder_cell.to(device)

model = EncoderDecoderWrapper(
    encoder,
    decoder_cell,
    output_size=90,
    teacher_forcing=0,
    sequence_len=180,
    decoder_input=True,
    device='cuda'
)

model = model.to(device)

encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-2)
decoder_optimizer = torch.optim.AdamW(decoder_cell.parameters(), lr=1e-3, weight_decay=1e-2)

encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader),
                                                  epochs=6)
decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader),
                                                  epochs=6)

model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

xb, yb = next(iter(train_dataloader))
xb = [xbi.to(device) for xbi in xb]
yb = yb.to(device)

trainer = TorchTrainer(
    'encdec_ohe_std_mse_wd1e-2_do2e-1_test_hs100_tf0_adam',
    model,
    [encoder_optimizer, decoder_optimizer],
    loss_function,
    [encoder_scheduler, decoder_scheduler],
    device,
    scheduler_batch_step=True,
    pass_y=True,
    # additional_metric_fns={'SMAPE': smape_exp_loss}
)

trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=50)

vd = valid_dataloader if mode == 'valid' else None
trainer.train(6, train_dataloader, vd, resume_only_model=True, resume=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cpu')
class DAEncoder(nn.Module):
    def __init__(self, da_input_size, da_embed_size, da_hidden):
        super(DAEncoder, self).__init__()
        self.hidden_size = da_hidden
        self.xe = nn.Embedding(da_input_size, da_embed_size)
        self.eh = nn.Linear(da_embed_size, da_hidden)

    def forward(self, DA):
        embedding = torch.tanh(self.eh(self.xe(DA))) # (batch_size, 1) -> (batch_size, 1, hidden_size)
        return embedding


class DAContextEncoder(nn.Module):
    def __init__(self, da_hidden):
        super(DAContextEncoder, self).__init__()
        self.hidden_size = da_hidden
        self.hh = nn.GRU(da_hidden, da_hidden, batch_first=True)

    def forward(self, input_hidden, prev_hidden):
        output = input_hidden
        output, hidden = self.hh(output, prev_hidden)
        return output, hidden

    def initHidden(self, batch_size):
        # h_0 = (num_layers * num_directions, batch_size, hidden_size)
        return torch.zeros(1, batch_size, self.hidden_size).to(device)


class DADecoder(nn.Module):
    def __init__(self, da_input_size, da_embed_size, da_hidden):
        super(DADecoder, self).__init__()
        self.he = nn.Linear(da_hidden, da_embed_size)
        self.ey = nn.Linear(da_embed_size, da_input_size)

    def forward(self, hidden):
        pred = self.ey(torch.tanh(self.he(hidden)))
        return pred


class UtteranceEncoder(nn.Module):
    def __init__(self, utt_input_size, embed_size, utterance_hidden, padding_idx, dropout=0.2, bidirectional=True, fine_tuning=False):
        super(UtteranceEncoder, self).__init__()
        self.hidden_size = utterance_hidden
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.xe = nn.Embedding(utt_input_size, embed_size)
        self.xe.weight.requires_grad = False if fine_tuning else True
        self.eh = nn.Linear(embed_size, utterance_hidden)
        self.dropout = nn.Dropout(p=dropout) 
        self.hh = nn.GRU(utterance_hidden, utterance_hidden, num_layers=1, batch_first=True, bidirectional=bidirectional)

    def forward(self, X, hidden):
        lengths = (X != self.padding_idx).sum(dim=1).cpu()
        seq_len, sort_idx = lengths.sort(descending=True)
        _, unsort_idx = sort_idx.sort(descending=False)
        # sorting
        X = torch.tanh(self.eh(self.xe(X))) # (batch_size, 1, seq_len, embed_size)
        sorted_X = X[sort_idx]
        # padding
        packed_tensor = pack_padded_sequence(sorted_X, seq_len, batch_first=True)
        output, hidden = self.hh(packed_tensor, hidden)
        # unpacking
        output, _ = pad_packed_sequence(output, batch_first=True)
        # unsorting
        output = output[unsort_idx]
        hidden = hidden[:, unsort_idx]
        return output, hidden

    def initHidden(self, batch_size):
        layers = 2 if self.bidirectional else 1
        return torch.zeros(layers, batch_size, self.hidden_size).to(device)


class UtteranceContextEncoder(nn.Module):
    def __init__(self, utterance_hidden_size):
        super(UtteranceContextEncoder, self).__init__()
        self.hidden_size = utterance_hidden_size
        self.hh = nn.GRU(utterance_hidden_size, utterance_hidden_size, batch_first=True)

    def forward(self, input_hidden, prev_hidden):
        output = input_hidden
        output, hidden = self.hh(output, prev_hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)


class UtteranceDecoder(nn.Module):
    def __init__(self, utterance_hidden_size, utt_embed_size, utt_vocab_size):
        super(UtteranceDecoder, self).__init__()
        self.hidden_size = utterance_hidden_size
        self.embed_size = utt_embed_size
        self.vocab_size = utt_vocab_size

        self.ye = nn.Embedding(self.vocab_size, self.embed_size)
        self.eh = nn.Linear(self.embed_size, self.hidden_size)
        self.hh = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.he = nn.Linear(self.hidden_size, self.embed_size)
        self.ey = nn.Linear(self.embed_size, self.vocab_size)
        self.th = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, Y, hidden, tag=None):
        h = torch.tanh(self.eh(self.ye(Y)))
        if not tag is None:
            h = self.th(torch.cat((h, tag), dim=2))
        output, hidden = self.hh(h, hidden)
        y_dist = self.ey(torch.tanh(self.he(output.squeeze(1))))
        return y_dist, hidden, output


class DAPairEncoder(nn.Module):
    def __init__(self, da_hidden_size, da_embed_size, da_vocab_size):
        super(DAPairEncoder, self).__init__()
        self.hidden_size = da_hidden_size
        self.embed_size = da_embed_size
        self.vocab_size = da_vocab_size

        self.te = nn.Embedding(self.vocab_size, self.embed_size)
        self.eh = nn.Linear(self.embed_size * 2, self.hidden_size)

    def forward(self, X):
        embeded = self.te(X)
        c, n, _, _ = embeded.size()
        embeded = embeded.view(c, n, -1)
        return torch.tanh(self.eh(embeded))


class OrderReasoningLayer(nn.Module):
    def __init__(self, encoder_hidden_size, hidden_size, da_hidden_size, attn=False):
        super(OrderReasoningLayer, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.da_hidden_size = da_hidden_size
        self.attn = attn

        self.xh = nn.Linear(self.encoder_hidden_size * 2, self.hidden_size)
        self.hh = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=False)
        self.hh_b = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=False)
        self.tt = nn.GRU(self.da_hidden_size, self.da_hidden_size)
        self.max_pooling = ChannelPool(kernel_size=3)
        self.attention = Attention(self.hidden_size)
        self.attention_b = Attention(self.hidden_size)

    def forward(self, X, DA, hidden, da_hidden):
        X = self.xh(X)
        output, hidden_f = self.hh(X, hidden)
        output_b, hidden_b = self.hh_b(self._invert_tensor(X), hidden)
        # output: (window_size, batch_size, hidden_size)

        if not DA is None:
            da_output, _ = self.tt(DA, da_hidden)
            da_output = da_output[-1]
        else:
            da_output = None

        if self.attn:
            output = output.permute(1, 0, 2)
            output_b = output_b.permute(1, 0, 2)
            attns = self.attention(output.contiguous())
            attns_b = self.attention_b(output_b.contiguous())
            output = (output * attns).sum(dim=1)
            output_b = (output_b * attns_b).sum(dim=1)
            output = torch.cat((output, output_b), dim=-1)
        else:
            output = torch.cat((self.max_pooling.forward(output), self.max_pooling.forward(output_b)), dim=1)
        # output: (batch_size, hidden_size * 2)
        return output, da_output

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

    def initDAHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.da_hidden_size).to(device)

    def _invert_tensor(self, X):
        return X[torch.arange(X.size(0)-1, -1, -1)]

class Classifier(nn.Module):
    def __init__(self, hidden_size, middle_layer_size, da_hidden_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.middle_layer_size = middle_layer_size
        self.da_hidden_size = da_hidden_size

        self.hm = nn.Linear(self.hidden_size, self.middle_layer_size)
        self.mm = nn.Linear(self.middle_layer_size + self.da_hidden_size, self.middle_layer_size)
        self.my = nn.Linear(self.middle_layer_size, 1)

    def forward(self, X, DA=None):
        if not DA is None:
            output = self.mm(torch.cat((self.hm(X), DA), dim=-1))
        else:
            output = self.hm(X)
        tmp = self.my(output)
        pred = torch.sigmoid(tmp)
        return pred

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(
            nn.Linear(self.hidden_size, 24),
            nn.ReLU(True),
            nn.Linear(24,1)
        )

    def forward(self, output):
        b_size = output.size(0)
        attn_ene = self.layer(output.view(-1, self.hidden_size))
        return F.softmax(attn_ene.view(b_size, -1), dim=-1).unsqueeze(-1)


class ChannelPool(nn.MaxPool1d):
    def forward(self, X):
        X = X.permute(1,2,0)
        pooled = F.max_pool1d(X, self.kernel_size)
        pooled = pooled.permute(2,0,1).squeeze(0)
        return pooled


class BeamNode(object):
    def __init__(self, hidden, previousNode, wordId, logProb, length):
        self.hidden = hidden
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.eval() < other.eval()

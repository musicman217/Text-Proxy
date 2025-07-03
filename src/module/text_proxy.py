import torch
import torch.nn.functional as F
from torch import nn
import copy
from src.utils.distributed import AllGather
allgather = AllGather.apply
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Proxy_Layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,  # 512, 1, 512
                 activation="relu", normalize_before=False, is_weights=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 512 -> 512
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 512 -> 512

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before
        self.is_weights = is_weights

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        tgt = self.norm1(tgt) # query：(a,b,512), K,V：(M,b,512)
        memory = self.norm2(memory)
        tgt2, atten_weights = self.multihead_attn(tgt, memory, memory,)  # tgt2: (num_proxy, B, 512), weights: (B, 1, 12, 12*12)
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm4(tgt)

        return tgt, atten_weights

class TransDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        output = tgt

        intermediate = []
        all_weights = []

        for layer in self.layers:
            output, weights = layer(output, memory,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                all_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(all_weights)
        return output.unsqueeze(0)

class Proxy_decoder(nn.Module):
    def __init__(self, layers=1, heads=1, dim_ftr=512, dim_feedforward=512):
        super().__init__()
        embedding_dim = dim_ftr  # 512
        d_model = dim_ftr
        dim_feedforward = dim_feedforward
        decoder_layer = Proxy_Layer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.event_decoder = TransDecoder(decoder_layer, layers, nn.LayerNorm(d_model),
                                          return_intermediate=True)


    def forward(self, query, features):
        batch_size = features.shape[0]  # b <=(b,M=4,512)
        dim_num = features.shape[2]

        enco_others = features.permute(1, 0, 2)  # (M,b,512)
        h_attr = query # (a,512)

        if len(h_attr.size()) == 2: # (a,512)
            h_attr = h_attr.unsqueeze(0).repeat(batch_size, 1, 1)  # -> (b,a,512)
            h_attr_batch = h_attr.permute(1,0,2)  # -> (a,b,512)
        else:
            h_attr_batch = h_attr.permute(1,0,2) # (a=b,1,512)->(1,a=b,512) or (b,a,512)

        hs, _ = self.event_decoder(h_attr_batch, enco_others)  # query：(a,b,512), K,V：(M,b,512)
        hs = hs[-1].permute(1, 0, 2) # -> (B, a, 512)

        return hs


class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(out_features)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x) + x
        x = self.norm2(x)

        return x


class TextProxy(nn.Module):
    """
    for text proxy generating
    """
    def __init__(self, config):
        super(TextProxy, self).__init__()
        self.cfg = config
        self.proxy_decoder = Proxy_decoder(layers=2, heads=1, dim_ftr=512, dim_feedforward=512)

        self.dash_factor = nn.Parameter(torch.rand(1), requires_grad=True)
        self.dash_weights = nn.Linear(self.cfg.using_M, 512)  # m->512


    def forward(self, text_feat, video_feat, frame_feat, is_train=False, step=None):
        """
            text: (a,512)
            video: (b,m,512)
            frame: (b,m,512)
        """
        if is_train:
            text_feat = allgather(text_feat.contiguous(), self.cfg)
            video_feat = allgather(video_feat.contiguous(), self.cfg)
            frame_feat = allgather(frame_feat.contiguous(), self.cfg)

        pro_proxies = self.proxy_decoder(text_feat, video_feat[:,:self.cfg.using_M,:])
        pro_proxies = pro_proxies.permute(1, 0, 2)  # ->(a,b,512)

        text_feat_ = text_feat / text_feat.norm(dim=-1, keepdim=True)
        frame_feat_ = frame_feat / frame_feat.norm(dim=-1, keepdim=True)
        dir_vec = self.cfg.delta * text_feat.unsqueeze(1) - self.cfg.eta * pro_proxies
        dir_vec = dir_vec / torch.norm(dir_vec, p=2, dim=2, keepdim=True) # (a,b,1)
        frame_logits = torch.matmul(text_feat_, frame_feat_.transpose(1, 2)).permute(1, 0, 2)
        # (a,512)x(b,512,f)->(b,a,f)->(a,b,f)

        if self.cfg.dash_option == "linear":
            dash = self.dash_weights(frame_logits)  # (a,b,f)->(a,b,512) or (a=b,1,f)->(a=b,1,512)
            dash = torch.exp(dash)
        elif self.cfg.dash_option == "theta":
            dash = torch.exp(self.dash_factor * frame_logits.mean(2)).unsqueeze(2)  # -> (a,b,1)

        text_proxy = text_feat.unsqueeze(1) + dash * dir_vec
        # (a,1,512)+(a,b,512)*(a,b,512)->(a,b,512)

        return text_proxy

import torch.nn as nn
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)
        #self.apply(init_weights)
    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))
        a = F.softmax(self.contx(u), dim=1)
        s = (a * inp).sum(1)
        return s


class TransformerLayer(nn.Module):
    def __init__(self,both=True,
                 pretrained_path='aubmindlab/bert-base-arabert'):
        super(TransformerLayer, self).__init__()

        self.both = both
        self.transformer = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)


    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #(output_last_layer, pooled_cls, (output_layers))
        #output[0] (8, seqlen=64, 768) cls [8, 768] ( 12 (8, seqlen=64, 768))

        return outputs

    def output_num(self):
        return self.transformer.config.hidden_size

class ATTClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.2):
        super(ATTClassifier, self).__init__()
        self.attention = AttentionWithContext(in_feature)

        self.Classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(768, class_num)
        )

        self.apply(init_weights)

    def forward(self, x):
        att = self.attention(x[0]) #(X[0] (bs, seqlenght, embedD) att = \sum_i alpha_i x[0][i]

        xx = att + x[1]

        out = self.Classifier(xx)
        return out


class CLSClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.2):
        super(CLSClassifier, self).__init__()

        self.Classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_feature, class_num)
        )

        self.apply(init_weights)

    def forward(self, x):

        out = self.Classifier(x[1])
        return out

class VHClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.15, nl=6):
        super(VHClassifier, self).__init__()
        self.nl = nl
        self.Hattention = AttentionWithContext(in_feature)
        self.Vattention = AttentionWithContext(in_feature)
        l = 12 - nl
        self.VMattention = nn.ModuleList([AttentionWithContext(in_feature) for K in range(l)])

        self.Classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(768, class_num)
        )

        self.apply(init_weights)

    def forward(self, x):
        vx = x[2][self.nl:12]
        att_h = self.Hattention(x[0])
        att_v = [att_layer(vx[k]).unsqueeze(1) for k, att_layer in enumerate(self.VMattention)] # list ( [768], [768])
        att_v = torch.cat(att_v, 1) # [bs ,6, 768]
        att_v = self.Vattention(att_v) # [bs, 768]

        att = att_h + att_v + x[1]

        out = self.Classifier(att)

        return out


class MTCLSClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.2):
        super(MTCLSClassifier, self).__init__()

        self.MisClassifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_feature, 1)
        )

        self.CatClassifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_feature, 8)
        )

        self.apply(init_weights)

    def forward(self, x):

        out_mis = self.MisClassifier(x[1])
        out_cat = self.CatClassifier(x[1])
        return out_mis, out_cat


class MTATTClassifier(nn.Module):
    def __init__(self, in_feature, class_num=8, dropout_prob=0.15):
        super(MTATTClassifier, self).__init__()
        self.misattention = AttentionWithContext(in_feature)
        self.catattention = AttentionWithContext(in_feature)
        self.dropoutmis = nn.Dropout(dropout_prob)
        self.dropoutcat = nn.Dropout(dropout_prob)


        self.MisClassifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 1)
        )

        self.CatClassifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, class_num)
        )
        self.apply(init_weights)

    def forward(self, x):
        att_mis = self.misattention(x[0])
        att_cat = self.catattention(x[0])


        x_mis = att_mis + x[1]
        x_mis = self.dropoutmis(x_mis)
        x_cat = att_cat + x[1]
        x_cat = self.dropoutcat(x_cat)


        out_mis = self.MisClassifier(x_mis)
        out_cat = self.CatClassifier(x_cat)

        return out_mis, out_cat


class VHMTClassifier(nn.Module):
    def __init__(self, in_feature, class_num=8, dropout_prob=0.15, nl=6):
        super(VHMTClassifier, self).__init__()
        self.nl = nl
        l = 12 - nl
        self.HMisattention = AttentionWithContext(in_feature)
        self.HCatattention = AttentionWithContext(in_feature)
        self.Vattention = AttentionWithContext(in_feature)
        self.VMattention = nn.ModuleList([AttentionWithContext(in_feature) for K in range(l)])
        self.dropoutmis = nn.Dropout(dropout_prob)
        self.dropoutcat = nn.Dropout(dropout_prob)

        self.MisClassifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 1)
        )

        self.CatClassifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, class_num)
        )

        self.apply(init_weights)

    def forward(self, x):
        vx = x[2][self.nl:12]
        att_hmis = self.HMisattention(x[0])
        att_hcat = self.HCatattention(x[0])
        att_v = [att_layer(vx[k]).unsqueeze(1) for k, att_layer in enumerate(self.VMattention)]
        att_v = torch.cat(att_v, 1)
        att_v = self.Vattention(att_v)

        att_mis = att_hmis + att_v + x[1]
        att_mis = self.dropoutmis(att_mis)
        att_cat = att_hcat + att_v + x[1]
        att_cat = self.dropoutcat(att_cat)

        out_mis = self.MisClassifier(att_mis)
        out_cat = self.CatClassifier(att_cat)

        return out_mis, out_cat





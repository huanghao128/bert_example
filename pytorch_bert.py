import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from tokenization import BertTokenizer


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Attention(nn.Module):
    """
    Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        # softmax得到概率得分p_atten,
        p_attn = F.softmax(scores, dim=-1)
        # 如果有 dropout 就随机 dropout 比例参数
        if dropout is not None:
            p_attn = dropout(p_attn)
        attn_output = torch.matmul(p_attn, value)

        return attn_output, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        # h 表示模型个数
        super().__init__()
        assert d_model % h == 0

        # d_k表示key长度，d_model表示模型输出维度，需保证为h的正数倍
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        #1 Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linear_layers, (query, key, value))]

        #2 Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        #3 Concat using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.ones(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = self.a * (x-mean) / (std + self.eps) + self.b
        return output


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, ff_hidden, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=ff_hidden, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layernorm1 = LayerNorm(hidden)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layernorm2 = LayerNorm(hidden)
        
    def forward(self, x, mask):
        attn_out = self.multihead_attention(x, x, x, mask=mask)
        out1 = self.layernorm1(x + self.dropout1(attn_out))
        ffn_out = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_out))
        return out2


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, maxlen=512, dropout_rate=0.1):
        super(BERTEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment_embeddings = nn.Embedding(2, embed_size, padding_idx=0)
        # self.position_embeddings = PositionalEmbedding(d_model=embed_size)
        self.position_embeddings = nn.Embedding(maxlen, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.embed_size = embed_size
    
    def forward(self, x, segment_label):
        token_embeddings = self.token_embeddings(x)
        segment_embeddings = self.segment_embeddings(segment_label)
        position_embeddings = self.position_embeddings(x)
        embeddings = token_embeddings + segment_embeddings + position_embeddings
        return self.dropout(embeddings)


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, maxlen=512, num_layers=12, attn_heads=12, dropout=0.1):
        super(BERT, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.attn_heads = attn_heads
        
        self.ff_hidden = hidden * 4
        self.embeddings = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, maxlen=maxlen)
        
        self.transformer_bolcks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.ff_hidden, dropout) for _ in range(num_layers)])
        
    def forward(self, x, segment_label):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embeddings(x, segment_label)
        
        for transformer in self.transformer_bolcks:
            x = transformer.forward(x, mask)
            
        return x


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def gather_index(self, inputs, indexs, dim=1):
        dummy = indexs.unsqueeze(2).expand(indexs.size(0), indexs.size(1), inputs.size(2))
        out = torch.gather(inputs, dim, dummy)
        return out
    
    def forward(self, x, positions):
        x = self.gather_index(x, positions)
        print(x.size())
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


class MaskedLanguageModel2(nn.Module):
    def __init__(self, bert_embedding_weights):
        super(MaskedLanguageModel2, self).__init__()

        self.linear = nn.Linear(bert_embedding_weights.size(1), bert_embedding_weights.size(0), bias=False)
        self.linear.weight = bert_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_embedding_weights.size(0)))

    def gather_index(self, inputs, indexs, dim=1):
        dummy = indexs.unsqueeze(2).expand(indexs.size(0), indexs.size(1), inputs.size(2))
        out = torch.gather(inputs, dim, dummy)
        return out
    
    def forward(self, x, positions):
        x = self.gather_index(x, positions)
        print(x.size())
        x = self.linear(x) + self.bias
        return F.log_softmax(x, dim=1)
    

class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden, 2)

    def forward(self, x): 
        x0 = self.linear(x[:,0])
        return F.log_softmax(x0, dim=1)


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    def __init__(self, vocab_size, hidden=768, maxlen=512, 
                num_layers=12, attn_heads=12, dropout=0.1):
        super(BERTLM, self).__init__()
        
        self.bert = BERT(vocab_size, hidden, maxlen, num_layers, attn_heads, dropout)
        self.next_sentence = NextSentencePrediction(hidden)
        self.mask_lm = MaskedLanguageModel(hidden, vocab_size)
    
    def forward(self, x, segment_label, masked_lm_positions):
        out = self.bert(x, segment_label)
        nsp_out = self.next_sentence(out)
        mlm_out = self.mask_lm(out, masked_lm_positions)
        print(nsp_out.size(), mlm_out.size())
        return nsp_out, mlm_out


def main():
    vocab_size = 10000 
    hidden = 768
    num_layers = 12
    attn_heads = 12
    dropout = 0.1
    learning_rate = 0.05
    epochs = 10

    model = BERTLM(vocab_size=vocab_size, hidden=hidden, num_layers=num_layers, attn_heads=attn_heads, dropout=dropout)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # 输入训练语料
        x = None
        segment_label = None
        label_next = None
        label_mask = None

        optimizer.zero_grad()

        # 传入模型
        next_sent_output, mask_lm_output = model(x, segment_label)
        next_sent_loss = criterion(next_sent_output, label_next)
        mask_loss = criterion(mask_lm_output.transpose(1,2), label_mask)

        loss = next_sent_loss + mask_loss

        loss.backward()
        optimizer.step()


if __name__ == '__main__':

    tokenizer = BertTokenizer("data/vocab.txt")
    vocab_size = len(tokenizer.vocab)

    model = BERTLM(vocab_size=vocab_size, hidden=768, maxlen=512, num_layers=2, attn_heads=12, dropout=0.1)
    print(model)

    summary(model, input_size=[(1,512), (1,512), (1,10)], depth=4, dtypes=[torch.long, torch.long, torch.long])
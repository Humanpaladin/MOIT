import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel
from bert import BertPreTrainedModel
from utils.nn import LSTM, Linear


class BertBiDAF(BertPreTrainedModel):
    def __init__(self, args, bert_config):
        super(BertBiDAF, self).__init__(bert_config)
        self.args = args

        if bert_config.tfm_mode == 'finetune':
            self.bert = BertModel(bert_config)

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(
            args.char_vocab_size,
            args.char_dim,
            padding_idx=1
        )
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.char_conv = nn.Sequential(
            nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
            nn.ReLU()
        )

        # 2. Word Embedding Layer
        self.word_emb = self.bert

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_q = Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(
            input_size=args.hidden_size * 8,
            hidden_size=args.hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout
        )

        self.modeling_LSTM2 = LSTM(
            input_size=args.hidden_size * 2,
            hidden_size=args.hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout
        )

        # 6. Output Layer
        self.start_idx_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.start_idx_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.end_idx_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.end_idx_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)

        self.output_LSTM = LSTM(input_size=args.hidden_size * 2,
                                hidden_size=args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):

            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)      # 200 -> 200
                g = getattr(self, 'highway_gate{}'.format(i))(x)        # 200 -> 200
                x = g * h + (1 - g) * x                                 # 200
            return x

        def att_flow_layer(c, q):       # att_flow_layer() 方法的
                                        #   输入:
                                        #       c: 为一个 batch 的 context 编码 ，(60, 373, 200)
                                        #       q: 为一个 batch 的 query 编码 ，  (60, 25, 200)
                                        #   输出:
                                        #       即为 论文中的 G (60, 373, 800)，即一个 batch 的每个单词 经由注意力层 处理后的 800 维的编码
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                qi = q.select(1, i).unsqueeze(1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            cq = torch.stack(cq, dim=-1)


            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            a = F.softmax(s, dim=2)
            c2q_att = torch.bmm(a, q)   # 这里 c2q_att 即为论文中 U~ 矩阵，为 (60, 373, 200)

            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            q2c_att = torch.bmm(b, c).squeeze()
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)        # 这里 q2c_att 即为论文中 H~ 矩阵，为 (60, 373, 200)

            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)   # 最后将对应于论文中的 4 个向量:     (均为 (60, 373, 200) 维)
                                                                            #   H
                                                                            #   U~
                                                                            #   H * U~
                                                                            #   H * H~
            return x


        def output_layer(g, m, l):      # g: 由 att_flow_layer 得到的 (60, 373, 800)
                                        # m: 由 modeling_layer 得到的 (60, 373, 200)
                                        # l: (60)
            # (batch, c_len)
            start_idx_logits = (self.start_idx_weight_g(g) + self.start_idx_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            end_idx_logits = (self.end_idx_weight_g(g) + self.end_idx_weight_m(m2)).squeeze()

            return start_idx_logits, end_idx_logits


        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])[0]
        q_word = self.word_emb(batch.q_word[0])[0]

        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]


        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c_word, c_lens))[0]           # c 为融合了上下文的 context 的各个单词的编码，为 (60, 373, 200)
        q = self.context_LSTM((q_word, q_lens))[0]           # q 为融合了上下文的 query 的各个单词的编码，为 (60, 25, 200)

        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)                        # 输入 c, q 分别为融合了上下的 context 和 query 的各个单词的编码，分别为 (60, 373, 200) 和 (60, 25, 200)
                                                        # 输出为 context 中每个句子的单词 经由注意力处理后的 编码表示，为 (60, 373, 800)

        # 5. Modeling Layer
        m1 = self.modeling_LSTM1((g, c_lens))[0]        # 输入为一个 tuple:
                                                        #   g:      (60, 373, 800)
                                                        #   c_lens: (60)
                                                        # 输出只看其第一个元素:
                                                        #   m1:     (60, 373, 200)
                                                        #   即 一个 batch 的 60 个句子的 373 个单词的 200 维的编码
        m = self.modeling_LSTM2((m1, c_lens))[0]        # 再将上面得到的 m1 放入另一个 LSTM 中进行处理，得到更新的 200 维的编码

        # 6. Output Layer
        start_idx_logits, end_idx_logits = output_layer(g, m, c_lens)

        return start_idx_logits, end_idx_logits



class BiDAF(nn.Module):
    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(
            args.char_vocab_size,
            args.char_dim,
            padding_idx=1
        )

        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
            nn.ReLU()
        )

        # 2. Word Embedding Layer
        self.word_emb = nn.Embedding.from_pretrained(
            pretrained,
            freeze=True
        )

        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_q = Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1)


        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(
            input_size=args.hidden_size * 8,
            hidden_size=args.hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout
        )

        self.modeling_LSTM2 = LSTM(
            input_size=args.hidden_size * 2,
            hidden_size=args.hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout
        )

        # 6. Output Layer
        self.start_idx_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.start_idx_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.end_idx_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.end_idx_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)

        self.output_LSTM = LSTM(input_size=args.hidden_size * 2,
                                hidden_size=args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)      # 200 -> 200
                g = getattr(self, 'highway_gate{}'.format(i))(x)        # 200 -> 200
                x = g * h + (1 - g) * x                                 # 200
            return x

        def att_flow_layer(c, q):       # att_flow_layer() 方法的
                                        #   输入:
                                        #       c: 为一个 batch 的 context 编码 ，(60, 373, 200)
                                        #       q: 为一个 batch 的 query 编码 ，  (60, 25, 200)
                                        #   输出:
                                        #       即为 论文中的 G (60, 373, 800)，即一个 batch 的每个单词 经由注意力层 处理后的 800 维的编码
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []                                             # cq 用来得到 论文里的 S 矩阵 (T, J)
            for i in range(q_len):
                qi = q.select(1, i).unsqueeze(1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            cq = torch.stack(cq, dim=-1)

            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            a = F.softmax(s, dim=2)     # 对论文中 S 的每一行进行 softmax()
            c2q_att = torch.bmm(a, q)   # 这里 c2q_att 即为论文中 U~ 矩阵，为 (60, 373, 200)

            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            q2c_att = torch.bmm(b, c).squeeze()
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)        # 这里 q2c_att 即为论文中 H~ 矩阵，为 (60, 373, 200)

            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)   # 最后将对应于论文中的 4 个向量:     (均为 (60, 373, 200) 维)
                                                                            #   H
                                                                            #   U~
                                                                            #   H * U~
                                                                            #   H * H~
            return x

        def output_layer(g, m, l):      # g: 由 att_flow_layer 得到的 (60, 373, 800)
                                        # m: 由 modeling_layer 得到的 (60, 373, 200)
                                        # l: (60)
            # (batch, c_len)
            start_idx_logits = (self.start_idx_weight_g(g) + self.start_idx_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            end_idx_logits = (self.end_idx_weight_g(g) + self.end_idx_weight_m(m2)).squeeze()

            return start_idx_logits, end_idx_logits

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)

        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)

        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]           # c 为融合了上下文的 context 的各个单词的编码，为 (60, 373, 200)
        q = self.context_LSTM((q, q_lens))[0]           # q 为融合了上下文的 query 的各个单词的编码，为 (60, 25, 200)

        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)                        # 输入 c, q 分别为融合了上下的 context 和 query 的各个单词的编码，分别为 (60, 373, 200) 和 (60, 25, 200)
                                                        # 输出为 context 中每个句子的单词 经由注意力处理后的 编码表示，为 (60, 373, 800)

        # 5. Modeling Layer
        m1 = self.modeling_LSTM1((g, c_lens))[0]        # 输入为一个 tuple:
                                                        #   g:      (60, 373, 800)
                                                        #   c_lens: (60)
                                                        # 输出只看其第一个元素:
                                                        #   m1:     (60, 373, 200)
                                                        #   即 一个 batch 的 60 个句子的 373 个单词的 200 维的编码
        m = self.modeling_LSTM2((m1, c_lens))[0]        # 再将上面得到的 m1 放入另一个 LSTM 中进行处理，得到更新的 200 维的编码

        # 6. Output Layer
        start_idx_logits, end_idx_logits = output_layer(g, m, c_lens)

        return start_idx_logits, end_idx_logits

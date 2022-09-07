
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1                                                       # tgt_seq_len:
                                                                                            #   tensor([10, 7]) -> tensor([9, 6])

        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)           # mask:
                                                                                            #   tensor([[False, False, False, False, False, False, False, False, False],
                                                                                            #           [False, False, False, False, False, False, True,  True,  True ]])   # (2, 9)

        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)                              # tgt_tokens:   (2, 9)
                                                                                            #   此时的 tgt_tokens 才为真正作为 gold 的序列:  第一个位置的 index (0) 不用预测，预测其余各个位置的 index; 并将最后一个 index (1) 之后补为 -100:
                                                                                            #       tensor([[   5,    7,   11,   11,    2,   15,   15,    2,    1],
                                                                                            #               [   6,    6,    9,    9,    2,    1, -100, -100, -100]])

        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))               # target:   (2, 9)
                                                                                            # input:    (2, 31, 9)
                                                                                            #
                                                                                            # 注，F.cross_entropy(input, target) 中的 input, target 的维度规定:
                                                                                            #   1. 当 input 为 2 维、target 为 1 维时，二者的维度可如:
                                                                                            #       input:  tensor(2, 5)
                                                                                            #       target: tensor(2)
                                                                                            #   2. 当 input 为 3 维、target 为 2 维时，二者的维度可如:
                                                                                            #       input:  tensor(2, 31, 9)
                                                                                            #       target: tensor(2, 9)
                                                                                            #      该种情况下与直觉认知不太符合，因此也有将二者分别降为 2 维和 1 维的情况

        return loss


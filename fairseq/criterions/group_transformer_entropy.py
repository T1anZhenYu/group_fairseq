# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
from torchvision.utils import save_image
import time
import torch
@register_criterion('group_transformer_entropy')
class GroupTransformerEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.len_pre_dim = args.length_pre_dim

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        loss, nll_loss,acc = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'acc': acc
        }
        
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        #lprobs,lprobs2,target2 = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        # add eos loss
        attns = net_output[1]['attn']
        loss_eos =  attns[:,:,-1][:,:-1].sum()
        loss_eos = loss_eos.float()
        

        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        # remove reduce
        '''
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        '''

        nll_loss = nll_loss.sum()+loss_eos

        smooth_loss = smooth_loss.sum()

        eps_i = self.eps / lprobs.size(-1)
    
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss 
        acc2 = 0


        lprobs2 = lprobs2.view(-1,lprobs2.size(-1))

        #restrict target2 in range(0~len_pre_dim)
        too_big = (target2 >=self.len_pre_dim)
        normal = (target2 < self.len_pre_dim)
        target2[normal] += int((target2[too_big]-self.len_pre_dim).sum()/normal.sum())
        target2[too_big]=self.len_pre_dim-1 

        nll_loss2 = -lprobs2.gather(dim=-1, index=target2)      
        smooth_loss2 = -lprobs2.sum(dim=-1,keepdim=True)
        if reduce:
            nll_loss2 = nll_loss2.sum()
            smooth_loss2 = smooth_loss2.sum()
        eps_i2 = self.eps / lprobs2.size(-1)

        loss2 = (1. - self.eps) * nll_loss2 + eps_i2 * smooth_loss2

        len_pre = torch.topk(lprobs2,1)[-1].squeeze(-1)

        acc1 = torch.eq(len_pre,target2).sum()/(len_pre.shape[0])
        
        len_pre = len_pre.view(net_output[1]['attn'].shape[0],-1)
        target2 = target2.view(net_output[1]['attn'].shape[0],-1)

        acc2 = float(torch.eq(len_pre.sum(dim=-1),target2.sum(dim=-1)).sum())/float((len_pre.shape[0]))
        loss_total = loss+loss2
        nll_loss_total = nll_loss2 + nll_loss
        

        return loss,nll_loss,acc2
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        acc = sum(log.get('acc',0) for log in logging_outputs)/len(logging_outputs)
        print('log acc should be ',acc,' ',len(logging_outputs))
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'acc':acc
        }

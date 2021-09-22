# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration code for AFL fuzzer."""

import shutil
import subprocess
import os

from fuzzers import utils

# OUT environment variable is the location of build directory (default is /out).

def prepare_build_environment():
    """Set environment variables used to build AFL-based fuzzers."""
    cflags = [
        '-O2', '-fno-omit-frame-pointer', '-gline-tables-only',
        '-fsanitize=address', '-fsanitize-coverage=trace-pc-guard'
    ]
    utils.append_flags('CFLAGS', cflags)
    utils.append_flags('CXXFLAGS', cflags)

    os.environ['CC'] = 'clang'
    os.environ['CXX'] = 'clang++'
    os.environ['FUZZER_LIB'] = '/libAFL.a'


def build():
    """Build fuzzer."""
    prepare_build_environment()
    utils.build_benchmark()


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

from fuzzers.randnn.pytorch_util import glorot_uniform, MLP


class AutoregSampler(nn.Module):
    def __init__(self, vocab, embed_dim):
        super(AutoregSampler, self).__init__()
        self.size_vocab = len(vocab)
        self.embed_dim = embed_dim
        self.out_pred = MLP(embed_dim, [embed_dim * 2, self.size_vocab])

    def policy(self, state):
        out_prob = F.softmax(self.out_pred(state), dim=-1)
        sampled = torch.multinomial(out_prob, 1)
        return sampled


class RnnSampler(AutoregSampler):
    def __init__(self, vocab, embed_dim):
        super(RnnSampler, self).__init__(vocab, embed_dim)

        self.init_h = nn.Parameter(torch.Tensor(1, embed_dim))
        self.init_c = nn.Parameter(torch.Tensor(1, embed_dim))
        self.token_embed = nn.Parameter(torch.Tensor(len(vocab), embed_dim))
        glorot_uniform(self)
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)

    def forward(self, num_samples, steps):
        cur_state = (self.init_h.repeat(num_samples, 1), self.init_h.repeat(num_samples, 1))

        samples = []
        for _ in range(steps):
            h, _ = cur_state
            sampled = self.policy(h)
            samples.append(sampled)
            embed_update = self.token_embed[sampled.view(-1)]
            cur_state = self.lstm(embed_update, cur_state)
        
        samples = torch.cat(samples, dim=-1)
        return samples


def fuzz(input_corpus, output_corpus, target_binary):
    """Run afl-fuzz on target."""

    seed_files = os.listdir(input_corpus)

    vocab = {}
    inv_map = {}
    samples = []
    for fname in seed_files:
        fname = os.path.join(input_corpus, fname)
        with open(fname, 'rb') as fin:
            content = fin.read()
            cur_sample = []
            for ch_byte in content:
                if not ch_byte in vocab:
                    val = len(vocab)
                    vocab[ch_byte] = val
                    inv_map[val] = ch_byte
                cur_sample.append(vocab[ch_byte])
        samples.append(cur_sample)

    nn_sampler = RnnSampler(vocab, 128)

    for idx in range(10000):
        inputs = nn_sampler(1, len(samples[0])).view(-1).data.cpu().numpy()

        fname = os.path.join(output_corpus, 'sample-%d' % idx)
        with open(fname, 'wb') as fout:
            arr = [inv_map[c] for c in inputs]
            arr = bytearray(arr)
            fout.write(arr)


if __name__ == '__main__':
    fuzz('./seeds', './corpus', './fuzz-target')
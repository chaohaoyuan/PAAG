from torchdrug import models

import os
from torch import nn
from torchdrug import core, layers

class ESM(models.ESM, nn.Module, core.Configurable):
    def __init__(self, path, model="ESM-1b", readout="mean"):
        nn.Module.__init__(self)
        core.Configurable.__init__(self)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        _model, alphabet = self.load_weight(path, model)
        self.alphabet = alphabet
        mapping = self.construct_mapping(alphabet)
        self.output_dim = self.output_dim[model]
        self.model = _model
        self.alphabet = alphabet
        self.repr_layer = self.num_layer[model]
        self.register_buffer("mapping", mapping)

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        elif readout == "pooler":
            self.readout = None
        else:
            raise ValueError("Unknown readout `%s`" % readout)
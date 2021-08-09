import torch
import torch.nn as nn
from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceToSequenceClassificationHead
from tape.registry import registry
from .modeling_utils import LayerNorm, MLMHead
from .modeling_bert import ProteinBertEncoder, ProteinBertForMaskedLM
from .modeling_lstm import ProteinLSTMEncoder, ProteinLSTMForLM
from .modeling_resnet import ProteinResNetEmbeddings, ProteinResNetForMaskedLM


class BottleneckConfig(ProteinConfig):
    def __init__(self,
                 hidden_size: int = 1024,
                 backend_name: str = 'resnet',
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.backend_name = backend_name


class BottleneckAbstractModel(ProteinModel):
    """ All your models will inherit from this one - it's used to define the
        config_class of the model set and also to define the base_model_prefix.
        This is used to allow easy loading/saving into different models.
    """
    config_class = BottleneckConfig
    base_model_prefix = 'bottleneck'

    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, ProteinResNetBlock):
            # nn.init.constant_(module.bn2.weight, 0)


@registry.register_task_model('pretraining', 'bottleneck')
class ProteinBottleneckForMaskedLM(BottleneckAbstractModel):
    
    def __init__(self, config):
        super().__init__(config)
        if config.backend_name == 'resnet':
            self.backbone1 = ProteinResNetEmbeddings()
            self.backbone2 = ProteinResNetForMaskedLM()
        elif config.backend_name == 'transformer':
            self.backbone1 = ProteinBertEncoder()
            self.backbone2 = ProteinBertForMaskedLM()
        elif config.backend_name == 'lstm':
            self.backbone1 = ProteinLSTMEncoder()
            self.backbone2 = ProteinLSTMForLM()
        else:
            raise ValueError('Somethings wrong')

        self.linear1 = nn.Linear(config.hidden_size, 2)
        self.linear2 = nn.Linear(2, config.hidden_size)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        out = self.backbone1(input_ids, input_mask, targets)
        embeddings = self.linear(out)
        outputs = self.backbone2(embeddings)

        return outputs
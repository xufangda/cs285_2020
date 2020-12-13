import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################
    def _get_action(self, obs:np.ndarray) -> torch.Tensor:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = torch.tensor(observation, requires_grad = True)

        # TODO return the action that the policy prescribes
        output_distribution=self(observation)
        if self.discrete:
            output = torch.argmax(output_distribution.sample())
        else:
            output=output_distribution.sample()

        # TODO return the action that the policy prescribes
        
        return output

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        output = self._get_action(obs)
        output = output.cpu().detach().numpy()

        return output

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if type(observation) is not torch.FloatTensor:
            observation = torch.tensor(observation, requires_grad=True, dtype=torch.float32)

        x = observation.to(ptu.device)
        if self.discrete:
            for layer in self.logits_na:
                x = layer(x)
            x=F.log_softmax(x).exp()
            output_distribution = torch.distributions.multinomial.Multinomial(total_count = 1, probs=x)
            return output_distribution
        else:
            for layer in self.mean_net:
                x = layer(x)    
            output_distribution = torch.distributions.normal.Normal(x, self.logstd)
            return output_distribution        
#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        
        self.optimizer.zero_grad()
        predicted_action = self._get_action(observations)
        actions = torch.tensor(actions, requires_grad= True).to(ptu.device)
        
        loss = self.loss(predicted_action, actions)
        loss.backward()

        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

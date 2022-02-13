import logging
import time
import numpy as np
from typing import List

from fl_main.lib.util.helpers import generate_model_id
from fl_main.lib.util.states import IDPrefix
from .state_manager import StateManager

class Aggregator:
    """
    Aggregator class instance provides a set of mathematical functions to compute the aggregated models.
    """
    def __init__(self, sm: StateManager):
        # state manager to access to models and model buffers
        self.sm = sm

    def _average_aggregate(self,
                           buffer: List[np.array],
                           num_samples: List[int]) -> np.array:
        """
        Given a list of models, compute the average model (FedAvg).
        This function provides a primitive mathematical operation.
        :param buffer: List[np.array] - A list of models to be aggregated
        :return: np.array - The aggregated models
        """
        denominator = sum(num_samples)
        # weighted average
        model = float(num_samples[0])/denominator * buffer[0]
        for i in range(1, len(buffer)):
            model += float(num_samples[i])/denominator * buffer[i]

        return model

    def aggregate_local_models(self):
        """
        Compute an average model for each tensor
        :return:
        """
        for mname in self.sm.mnames:
            self.sm.cluster_models[mname][0] \
                = self._average_aggregate(self.sm.local_model_buffers[mname], self.sm.local_model_num_samples)

        # Save the number of samples used
        self.sm.own_cluster_num_samples = sum(self.sm.local_model_num_samples)

        logging.info(f'--- Cluster models are formed ---')
        logging.debug(f'{self.sm.cluster_models}')

        # Create model ID
        id = generate_model_id(IDPrefix.aggregator, self.sm.id, time.time())
        self.sm.cluster_model_ids.append(id)

        # Clear buffered local models
        self.sm.clear_lmodel_buffers()
        logging.debug('Local model buffers cleared')

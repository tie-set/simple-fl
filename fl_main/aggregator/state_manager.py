import numpy as np
import logging
import time
from typing import Dict, Any

from fl_main.lib.util.data_struc import LimitedDict
from fl_main.lib.util.helpers import generate_id, generate_model_id
from fl_main.lib.util.states import IDPrefix

class StateManager:
    """
    StateManager class instance keeps the state of an aggregator.
    Every volatile state of an aggregator should be stored in this instance.
    The data includes:
    - local models delivered by agents
    - cluster models pulled from database
    - round number
    - agents under this aggregator
    - global cluster model
    - cluster model
    """

    def __init__(self):
        # unique ID for the aggregator
        self.id = generate_id()

        # informatioin of connected agents
        self.agent_set = list()

        # model names of ML models
        self.mnames = list()

        # aggregation round
        self.round = 0

        # stores cluster/local models by names
        # {'model_name' : a list of a type of models}
        self.local_model_buffers = LimitedDict(self.mnames)

        # stores sample numbers for each agent
        self.local_model_num_samples = list()

        # stores cluster models by names
        # {'model_name' : list of a type of models (only used location 0)}
        self.cluster_models = LimitedDict(self.mnames)

        # stores Model IDs of all models created by this aggregator
        self.cluster_model_ids = list()

        # State of the aggregator
        self.initialized = False

        # Aggregation threshold to be used for aggregation criteria
        self.agg_threshold = 1

    def ready_for_local_aggregation(self) -> bool:
        """
        Return a bool val to identify if it can starts the aggregation process
        :return: (boolean) True if it has enough local models to aggregate
            False otherwise. The threshold is configured in the JSON config.json
        """
        if len(self.mnames) == 0:
            return False

        num_agents = int(self.agg_threshold * len(self.agent_set))
        if num_agents == 0: num_agents = 1
        logging.info(f'--- Aggregation Threshold (Number of agents needed for aggregation): {num_agents} ---')

        num_collected_lmodels = len(self.local_model_buffers[self.mnames[0]])
        logging.info(f'--- Number of collected local models: {num_collected_lmodels} ---')

        if num_collected_lmodels >= num_agents:
            logging.info(f'--- Enough local models are collected. Aggregation will start. ---')
            return True
        else:
            logging.info(f'--- Waiting for more local models to be collected ---')
            return False

    def initialize_model_info(self, lmodels, init_weights_flag):
        """
        Initialize the structure of NNs (numpy.array) based on the first models received
        :param models: lmodels - local ML models
        :param models: init_weights_flag - for initializing base model
        :return:
        """
        for key in lmodels.keys():
            self.mnames.append(key)
        # print("model names:", self.mnames)
        self.local_model_buffers = LimitedDict(self.mnames)
        self.cluster_models = LimitedDict(self.mnames)

        # Clear all models saved and buffered
        self.clear_lmodel_buffers()

        if init_weights_flag:
            # Use the received local models as the cluster model (init base models)
            self.initialize_models(lmodels, weight_keep=init_weights_flag)
        else:
            # initialize the model with zeros
            self.initialize_models(lmodels, weight_keep=False)

    def initialize_models(self, models: Dict[str, np.array], weight_keep: bool = False):
        """
        Initialize the structure of NNs (numpy.array) based on the first models received
        :param models: Dict[str, np.array]
        :return:
        """
        self.clear_saved_models()
        for mname in self.mnames:
            if weight_keep:
                m = models[mname]
            else:
                # Keep the weights given as init weights
                m = np.zeros_like(models[mname])
            # Create cluster model and its ID
            self.cluster_models[mname].append(m)
            id = generate_model_id(IDPrefix.aggregator, self.id, time.time())
            self.cluster_model_ids.append(id)

        self.initialized = True  # set True so that it will never be called automatically
        logging.info(f'--- Model Formats initialized, model names: {self.mnames} ---')

    def buffer_local_models(self,
                            models: Dict[str, np.array],
                            participate=False,
                            meta_data: Dict[Any, Any] = {}):
        """
        Store a set of local models from an agent to the local model buffer
        :param meta_data: Meta info including num of samples
        :param models: Dict[str, np.array]
        :return:
        """
        if not participate:  # if it is an actual models (not in participation message)
            for key, model in models.items():
                self.local_model_buffers[key].append(model)

            try:
                # if num_samples is specified by the agent
                num_samples = meta_data["num_samples"]
            except:
                num_samples = 1

            self.local_model_num_samples.append(int(num_samples))
        else:  # if it comes from the participation message
            pass

        # if the cluster models have not been initialized
        # first time call only
        if not self.initialized:
            self.initialize_models(models)

    def clear_saved_models(self):
        """
        Clear all models stored for a next round (cluster models)
        :return:
        """
        for mname in self.mnames:
            self.cluster_models[mname].clear()

    def clear_lmodel_buffers(self):
        """
        Clear all buffered local models for a next round
        :return:
        """
        for mname in self.mnames:
            self.local_model_buffers[mname].clear()
        self.local_model_num_samples = list()

    def add_agent(self, agent_name: str, agent_id: str, agent_ip: str, socket: str):
        """
        Save the info of an agent
        :param agent_name: str - agent name
        :param agent_id: str - agent ID
        :param agent_ip: str - agent IP address
        :param socket: str - port number to the agent
        :return: agent_id, socket
        """
        for agent in self.agent_set:
            if agent_name == agent['agent_name']:
                print(f'{agent_name} already exists.')
                return  agent['agent_id'], agent['socket']

        agent = {
            'agent_name': agent_name,
            'agent_id': agent_id,
            'agent_ip': agent_ip,
            'socket': socket
        }
        self.agent_set.append(agent)
        return agent_id, socket

    def increment_round(self):
        """
        Increment the round number (called after each global model synthesis)
        :return:
        """
        self.round += 1
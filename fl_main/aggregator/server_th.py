import asyncio
import logging
import pickle
import sys
import time
import numpy as np
import websockets
from typing import List, Dict, Any

from fl_main.lib.util.data_struc import convert_LDict_to_Dict
from fl_main.lib.util.helpers import read_config, set_config_file
from fl_main.lib.util.messengers import generate_db_push_message, \
     generate_cluster_model_dist_message, generate_agent_participation_confirmation_message
from fl_main.lib.util.states import ParticipateMSGLocation, ModelType, AggMsgType

from .state_manager import StateManager
from .aggregation import Aggregator


class Server:
    """
    Server class instance provides interface between the aggregator and DB,
    and the aggregator and an agent (client)
    """

    def __init__(self):
        """
        Instantiation of a Server instance
        """
        # read the config file
        config_file = set_config_file("aggregator")
        self.config = read_config(config_file)

        # functional components
        self.sm = StateManager()
        self.agg = Aggregator(self.sm)  # aggregation functions

        # Set up FL server's IP address
        self.aggr_ip = self.config['aggr_ip']

        # port numbers, websocket info
        self.reg_socket = self.config['reg_socket']
        self.recv_socket = self.config['recv_socket']
        self.exch_socket = self.config['exch_socket']

        # Set up DB info to connect with DB
        self.db_ip = self.config['db_ip']
        self.db_socket = self.config['db_socket']
        self.db_ws = f'ws://{self.db_ip}:{self.db_socket}'

        # thresholds
        self.round_interval = self.config['round_interval']
        self.sm.agg_threshold = self.config['aggregation_threshold']


    async def register(self, websocket: str, path):
        """
        Receiving the participation message specifying the model structures
        Sending back socket information for future model exchanges.
        Sending back the welcome message as a response.
        :param websocket:
        :param path:
        :return:
        """
        # Receiving participation messages
        msg = pickle.loads(await websocket.recv())
        logging.info('--- Participate Message Received ---')
        logging.debug(f'Message: {msg}')

        # Check if it is a simulation run
        if msg[int(ParticipateMSGLocation.sim_flag)]:
            logging.info(f'--- This run is a simulation ---')
            es = msg[int(ParticipateMSGLocation.exch_socket)]
        else:
            es = self.exch_socket

        # Add an agent to the agent list
        addr = msg[int(ParticipateMSGLocation.agent_ip)]
        agent_ws = f'ws://{addr}:{es}'
        self.sm.add_agent(agent_ws)

        # send back 'welcome' message with socket information for future model exchanges
        reply = generate_agent_participation_confirmation_message(
            f'{self.sm.round}', f'{es}', f'{self.recv_socket}'
        )

        # send the message
        await websocket.send(pickle.dumps(reply))

        # wait for sending messages
        await asyncio.sleep(1)

        # If the weights in the first models should be used as the init models
        # The very first agent connecting to the aggregator decides the shape of the models
        if self.sm.round == 0:

            # push local model info to DB
            agent_id = msg[int(ParticipateMSGLocation.agent_id)]
            model_id = msg[int(ParticipateMSGLocation.model_id)]
            gene_time = msg[int(ParticipateMSGLocation.gene_time)]
            lmodels = msg[ParticipateMSGLocation.lmodels] # <- Extract local models
            performance = msg[int(ParticipateMSGLocation.meta_data)]
            await self._push_local_models(agent_id, model_id, lmodels, gene_time, performance)

            # Clear all models saved and buffered
            self.sm.clear_lmodel_buffers()

            init_weights_flag = bool(msg[int(ParticipateMSGLocation.init_flag)])
            if init_weights_flag:
                # Use the received local models as the cluster model (init base models)
                self.sm.initialize_models(lmodels, weight_keep=init_weights_flag)
            else:
                # initialize the model with zeros
                self.sm.initialize_models(lmodels, weight_keep=False)

            # wait for sending messages
            await asyncio.sleep(1)

            # Send out the cluster models
            await self._send_cluster_models_to_all()

            # Recognize this step as one aggregation round
            self.sm.increment_round()

        # If there was at least one SG aggregation
        if self.sm.round > 0:
            # send cluster models to the agent
            model_id = self.sm.cluster_model_ids[-1]
            cluster_models = convert_LDict_to_Dict(self.sm.cluster_models)

            msg = generate_cluster_model_dist_message(self.sm.id, model_id, self.sm.round, cluster_models)
            async with websockets.connect(agent_ws, max_size=None, max_queue=None) as websocket:
                await self._send_cluster_models_to_one_agent(pickle.dumps(msg), websocket)


    async def receive_local_models(self, websocket: str, path):
        """
        Receiving local model updates
        :param websocket:
        :param path:
        :return:
        """
        um = await websocket.recv()
        msg = pickle.loads(um)

        print("received")
        lmodels = msg[int(ParticipateMSGLocation.lmodels)]
        agent_id = msg[int(ParticipateMSGLocation.agent_id)]
        model_id = msg[int(ParticipateMSGLocation.model_id)]
        gene_time = msg[int(ParticipateMSGLocation.gene_time)]
        performance = msg[int(ParticipateMSGLocation.meta_data)]
        await self._push_local_models(agent_id, model_id, lmodels, gene_time, performance)

        logging.info('--- Local Model Received ---')
        logging.debug(f'Local models: {lmodels}')

        # Store local models in the buffer
        self.sm.buffer_local_models(lmodels, participate=False, meta_data=performance)


    async def model_synthesis_routine(self):
        """
        Periodically check the number of stored models and
         execute synthesis if there are enough based on the agreed threshold
        :return:
        """
        while True:
            # Periodic check (frequency is specified in the JSON config file)
            await asyncio.sleep(self.round_interval)

            if self.sm.ready_for_local_aggregation():  # if it has enough models to aggregate
                logging.info(f'Round {self.sm.round}')
                logging.info(f'Current agents: {self.sm.agent_set}')

                # --- Local aggregation process --- #
                # Local models --> An cluster model #
                # Create a cluster model from local models
                self.agg.aggregate_local_models()

                # Push cluster model to DB
                await self._push_cluster_models()

                await self._send_cluster_models_to_all()

                # increment the aggregation round number
                self.sm.increment_round()


    async def _send_cluster_models_to_one_agent(self, bsgms, websocket):
        """
        Send a binary file (message) to an agent through a give websocket
        :param bsgms: Message (binary file)
        :param websocket:
        :return:
        """
        while not websocket:  # wait until socket being initialized
            await asyncio.sleep(0.001)
        await websocket.send(bsgms)

    async def _send_cluster_models_to_all(self):
        """
        Send out cluster models to all agents under this aggregator
        :return:
        """
        model_id = self.sm.cluster_model_ids[-1]
        cluster_models = convert_LDict_to_Dict(self.sm.cluster_models)

        msg = generate_cluster_model_dist_message(self.sm.id, model_id, self.sm.round, cluster_models)
        bsgms = pickle.dumps(msg)

        for agent in self.sm.agent_set:
            try:
                async with websockets.connect(agent, max_size=None, max_queue=None) as websocket:
                    await self._send_cluster_models_to_one_agent(bsgms, websocket)
                logging.info(f'--- Cluster Models Sent to {agent}---')
            except:
                logging.error("connection lost to the agent: " + agent)
                logging.error(f'--- Cluster Models NOT Sent ---')

    async def _push_local_models(self, agent_id: str, model_id: str, local_models: Dict[str, np.array],\
                                 gene_time: float, performance: Dict[str, float]) -> List[Any]:
        """
        Pushing a given set of local models to DB
        :param agent_id: str - ID of the agent that created this local model
        :param model_id: str - Model ID passed from the agent
        :param local_models: Dict[str,np.array] - Local models
        :param gene_time: float - the time at which the models were generated
        :param performance: Dict[str,float] - Each entry is a pair of model ID and its performance metric
        :return: Response message (List)
        """
        logging.debug(f'The local models to send: {local_models}')
        return await self._push_models(agent_id, ModelType.local, local_models, model_id, gene_time, performance)

    async def _push_cluster_models(self) -> List[Any]:
        """
        Pushing the cluster models to DB
        :return: Response message (List)
        """
        logging.debug(f'My cluster models to send: {self.sm.cluster_models}')
        model_id = self.sm.cluster_model_ids[-1]  # the latest ID
        models = convert_LDict_to_Dict(self.sm.cluster_models)
        meta_dict = dict({"num_samples" : self.sm.own_cluster_num_samples})
        return await self._push_models(self.sm.id, ModelType.cluster, models, model_id, time.time(), meta_dict)

    async def _push_models(self,
                           component_id: str,
                           model_type: ModelType,
                           models: Dict[str, np.array],
                           model_id: str,
                           gene_time: float,
                           performance_dict: Dict[str, float]) -> List[Any]:
        """
        Push a given set of models to DB
        :param component_id:
        :param models: LimitedDict - models
        :param model_type: model type
        :param model_id: str - model ID
        :param gene_time: float - the time at which the models were generated
        :param performance_dict: Dict[str, float] - Each entry is a pair of model id and its performance metric
        :return: Response message (List)
        """
        push_msg = generate_db_push_message(component_id, self.sm.round, model_type, models, model_id, gene_time, performance_dict)

        async with websockets.connect(self.db_ws, max_size=None, max_queue=None) as websocket:
            await websocket.send(pickle.dumps(push_msg))
            resp = pickle.loads(await websocket.recv())
            logging.info(f'--- Models pushed to DB: Response {resp} ---')

        return resp


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    s = Server()
    logging.info("--- Aggregator Started ---")

    loop = asyncio.get_event_loop()

    start_server = websockets.serve(s.register, s.aggr_ip, s.reg_socket,
                                    max_size=None, max_queue=None)

    start_receiver = websockets.serve(s.receive_local_models, s.aggr_ip, s.recv_socket,
                                      max_size=None, max_queue=None)

    loop.run_until_complete(asyncio.gather(start_server,
                                           start_receiver,
                                           s.model_synthesis_routine()))

    loop.run_forever()

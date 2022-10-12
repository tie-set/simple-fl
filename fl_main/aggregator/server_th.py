import asyncio, logging, time, numpy as np
from typing import List, Dict, Any

from fl_main.lib.util.communication_handler import init_fl_server, send, send_websocket, receive 
from fl_main.lib.util.data_struc import convert_LDict_to_Dict
from fl_main.lib.util.helpers import read_config, set_config_file
from fl_main.lib.util.messengers import generate_db_push_message, generate_ack_message, \
     generate_cluster_model_dist_message, generate_agent_participation_confirm_message
from fl_main.lib.util.states import ParticipateMSGLocation, ModelUpMSGLocation, PollingMSGLocation, \
     ModelType, AgentMsgType

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

        # thresholds
        self.round_interval = self.config['round_interval']
        self.sm.agg_threshold = self.config['aggregation_threshold']

        self.is_polling = bool(self.config['polling'])


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
        msg = await receive(websocket)
        logging.info(f'--- {msg[int(ParticipateMSGLocation.msg_type)]} Message Received ---')
        logging.debug(f'Message: {msg}')

        # Check if it is a simulation run
        es = self._get_exch_socket(msg)

        # Add an agent to the agent list
        agent_name = msg[int(ParticipateMSGLocation.agent_name)]
        agent_id = msg[int(ParticipateMSGLocation.agent_id)]
        addr = msg[int(ParticipateMSGLocation.agent_ip)]

        uid, ues = self.sm.add_agent(agent_name, agent_id, addr, es)

        # If the weights in the first models should be used as the init models
        # The very first agent connecting to the aggregator decides the shape of the models
        if self.sm.round == 0:
            await self._initialize_fl(msg)

        # If there was at least one global model, just proceed

        # Wait for sending messages
        await asyncio.sleep(0.5)

        # send back 'welcome' message
        await self._send_updated_global_model(websocket, uid, ues)

    def _get_exch_socket(self, msg):
        """
        Get EXCH Socket
        :param msg: Message received
        :return: exch_socket
        """
        if msg[int(ParticipateMSGLocation.sim_flag)]:
            logging.info(f'--- This run is a simulation ---')
            es = msg[int(ParticipateMSGLocation.exch_socket)]
        else:
            es = self.exch_socket
        return es

    async def _initialize_fl(self, msg):
        """
        Initialize FL round
        :param msg: Message received
        :return:
        """
        # Extract values from the message received
        agent_id = msg[int(ParticipateMSGLocation.agent_id)]
        model_id = msg[int(ParticipateMSGLocation.model_id)]
        gene_time = msg[int(ParticipateMSGLocation.gene_time)]
        lmodels = msg[int(ParticipateMSGLocation.lmodels)] # <- Extract local models
        performance = msg[int(ParticipateMSGLocation.meta_data)]
        init_weights_flag = bool(msg[int(ParticipateMSGLocation.init_flag)])

        # Initialize model info
        self.sm.initialize_model_info(lmodels, init_weights_flag)

        # Pushing the local model to DB
        await self._push_local_models(agent_id, model_id, lmodels, gene_time, performance)

        # Recognize this step as one aggregation round
        self.sm.increment_round()

    async def _send_updated_global_model(self, websocket, agent_id, exch_socket):
        """
        Send cluster models to the agent
        :param addr: IP address of agent
        :param es: Port of the agent
        :return:
        """
        model_id = self.sm.cluster_model_ids[-1]
        cluster_models = convert_LDict_to_Dict(self.sm.cluster_models)
        reply = generate_agent_participation_confirm_message(
            self.sm.id, model_id, cluster_models, 
            self.sm.round, agent_id, exch_socket, self.recv_socket)
        await send_websocket(reply, websocket)
        logging.info(f'--- Global Models Sent to {agent_id} ---')

    async def receive_msg_from_agent(self, websocket, path):
        """
        Receiving messages from agents for model updates or polling
        :param websocket:
        :param path:
        :return:
        """
        msg = await receive(websocket)

        if msg[int(ModelUpMSGLocation.msg_type)] == AgentMsgType.update:
            await self._process_lmodel_upload(msg)

        elif msg[int(PollingMSGLocation.msg_type)] == AgentMsgType.polling:
            await self._process_polling(msg, websocket)

    async def _process_lmodel_upload(self, msg):
        """
        Process local models uploaded from agents
        :param msg: message received from the agent
        :return:
        """
        lmodels = msg[int(ModelUpMSGLocation.lmodels)]
        agent_id = msg[int(ModelUpMSGLocation.agent_id)]
        model_id = msg[int(ModelUpMSGLocation.model_id)]
        gene_time = msg[int(ModelUpMSGLocation.gene_time)]
        perf_val = msg[int(ModelUpMSGLocation.meta_data)]
        await self._push_local_models(agent_id, model_id, lmodels, gene_time, perf_val)

        logging.info('--- Local Model Received ---')
        logging.debug(f'Local models: {lmodels}')

        # Store local models in the buffer
        self.sm.buffer_local_models(lmodels, participate=False, meta_data=perf_val)

    async def _process_polling(self, msg, websocket):
        """
        Process the polling message from agents
        :param msg: message received from the agent
        :param websocket:
        :return:
        """
        logging.debug(f'--- AgentMsgType.polling ---')
        # print(f'current round:', self.sm.round)
        # print(f'reported round:', str(msg[int(PollingMSGLocation.round)]))
        if self.sm.round > int(msg[int(PollingMSGLocation.round)]):
            model_id = self.sm.cluster_model_ids[-1]
            cluster_models = convert_LDict_to_Dict(self.sm.cluster_models)
            gm_msg = generate_cluster_model_dist_message(self.sm.id, model_id, self.sm.round, cluster_models)
            await send_websocket(gm_msg, websocket)
            logging.info(f'--- Global Models Sent to {msg[int(PollingMSGLocation.agent_id)]} ---')
        else:
            logging.info(f'--- Polling: Global model is not ready yet ---')
            msg = generate_ack_message()
            await send_websocket(msg, websocket)

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
                
                if self.is_polling == False:
                    await self._send_cluster_models_to_all()

                # increment the aggregation round number
                self.sm.increment_round()

    async def _send_cluster_models_to_all(self):
        """
        Send out cluster models to all agents under this aggregator
        :return:
        """
        model_id = self.sm.cluster_model_ids[-1]
        cluster_models = convert_LDict_to_Dict(self.sm.cluster_models)

        msg = generate_cluster_model_dist_message(self.sm.id, model_id, self.sm.round, cluster_models)
        for agent in self.sm.agent_set:
            await send(msg, agent['agent_ip'], agent['socket'])
            logging.info(f'--- Global Models Sent to {agent["agent_id"]} ---')

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
        msg = generate_db_push_message(component_id, self.sm.round, model_type, models, model_id, gene_time, performance_dict)
        resp = await send(msg, self.db_ip, self.db_socket)
        logging.info(f'--- Models pushed to DB: Response {resp} ---')

        return resp


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    s = Server()
    logging.info("--- Aggregator Started ---")

    init_fl_server(s.register, 
                   s.receive_msg_from_agent, 
                   s.model_synthesis_routine(), 
                   s.aggr_ip, s.reg_socket, s.recv_socket)

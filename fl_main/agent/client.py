import asyncio
import websockets
import time
import pickle
import logging
import sys
import os
from typing import Dict, List, Any

from threading import Thread
from fl_main.lib.util.helpers import read_config, \
save_model_file, load_model_file, read_state, write_state, generate_id, \
set_config_file, get_ip, compatible_data_dict_read, generate_model_id

# Client states
from fl_main.lib.util.states import ClientState, AgentMsgType, GMDistributionMsgLocation, IDPrefix
from fl_main.lib.util.messengers import generate_lmodel_update_message, generate_agent_participation_message

class Client:
    """
    Client class instance provides the communication interface
    between Agent's ML logic and an aggregator
    """

    def __init__(self):

        time.sleep(2)
        logging.info(f"Agent initilized at {time.time()}")

        # Unique ID in the system
        self.id = generate_id()

        self.agent_ip = get_ip()

        # Check command line argvs
        self.simulation_flag = False
        if len(sys.argv) > 1:
            # if sys.argv[1] == '1', it's in simulation mode
            self.simulation_flag = bool(int(sys.argv[1]))

        # Read config
        config_file = set_config_file("agent")
        self.config = read_config(config_file)

        # Comm. info to join the FL platform
        self.aggr_ip = self.config['aggr_ip']
        self.wsprefix = f'ws://{self.aggr_ip}:'
        self.msend_socket = 0  # later updated based on welcome message
        self.reg_socket = self.config['reg_socket']

        if self.simulation_flag:
            # if it's simulation, use the manual socket number
            # self.reg_socket = int(sys.argv[2])
            self.exch_socket = int(sys.argv[2])
        else:  # Not in simulation mode
            # self.reg_socket = self.config['reg_socket']
            self.exch_socket = 0  # later updated based on welcome message

        # Local file locations
        if self.simulation_flag:
            self.model_path = f'{self.config["model_path"]}/{sys.argv[3]}'
        else:
            self.model_path = f'{self.config["model_path"]}/default-agent'
            # self.model_path = self.config['model_path']

        # if there is no directory to save models
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.lmfile = self.config['local_model_file_name']
        self.gmfile = self.config['global_model_file_name']
        self.statefile = self.config['state_file_name']

        # Aggregation round - later updated by the info from the aggregator
        self.round = 0

        # State indicator
        self.waiting_flag = ClientState.training
        
        # Initialization
        self.init_weights_flag = bool(self.config['init_weights_flag'])

        # Importing ClientState
        self.ClientState = ClientState


    async def participate(self):
        """
        Send the first message to join an aggregator and
        Receive state/comm. info from the aggregator
        :return:
        """
        # Read the local models to tell the structure to the aggregator
        # (not necessarily trained)
        data_dict, performance_dict = load_model_file(self.model_path, self.lmfile)

        _, gene_time, models, model_id = compatible_data_dict_read(data_dict)

        logging.debug(models)

        wsaddr = f'{self.wsprefix}{self.reg_socket}'
        async with websockets.connect(wsaddr, max_size=None, max_queue=None) as websocket:
            # Create a participation message
            msg = generate_agent_participation_message(
                self.id, model_id, models, self.init_weights_flag, self.simulation_flag,
                self.exch_socket, gene_time, performance_dict, self.agent_ip)

            await websocket.send(pickle.dumps(msg))

            resp = pickle.loads(await websocket.recv())
            # Parse the response message
            # including some socket info and the actual round number
            self.round = resp[1]
            self.exch_socket = resp[2]
            self.msend_socket = resp[3]
            logging.debug(f"Init Response: {resp}")

        # State transition to waiting_gm
        self.tran_state(ClientState.waiting_gm)


    async def wait_models(self, websocket, path):
        """
        Waiting for cluster models from the aggregator
        :param websocket:
        :return:
        """
        gm_msg = pickle.loads(await websocket.recv())
        logging.info(f'--- Cluster Global Model Received ---')

        logging.debug(f'Models: {gm_msg}')

        # pass (model_id, models) to an app
        data_dict = dict()
        data_dict['model_id'] = gm_msg[int(GMDistributionMsgLocation.model_id)]
        data_dict['models'] = gm_msg[int(GMDistributionMsgLocation.global_models)]
        self.round = gm_msg[int(GMDistributionMsgLocation.round)]

        # Save the received cluster global models to the local file
        save_model_file(data_dict, self.model_path, self.gmfile)
        
        # State transition to sg_ready
        self.tran_state(ClientState.gm_ready)


    async def model_exchange_routine(self):
        """
        Check the progress of training and send the updated models
        once the training is done
        :return:
        """
        while True:
            # Periodically check the state
            await asyncio.sleep(3)
            state = read_state(self.model_path, self.statefile)

            if state == ClientState.sending: # ready to send
                # Read the models from the local file
                data_dict, performance_dict = load_model_file(self.model_path, self.lmfile)

                _, _, models, model_id = compatible_data_dict_read(data_dict)
                upd_msg = generate_lmodel_update_message(self.id, model_id, models, performance_dict)

                wsadddr = f'{self.wsprefix}{self.msend_socket}'
                async with websockets.connect(wsadddr, max_size=None, max_queue=None) as websocket:
                    await websocket.send(pickle.dumps(upd_msg))

                # State transition to waiting_gm
                self.tran_state(ClientState.waiting_gm)
                logging.info('--- Local Models Sent ---')
            elif state == ClientState.training:
                # Do nothing
                logging.info(f'--- Training is happening ---')
                await asyncio.sleep(3)
            elif state == ClientState.gm_ready:
                # Do nothing
                logging.info(f'--- Global Model is ready ---')
                await asyncio.sleep(3)


    # Starting FL client functions
    def start_fl_client(self):
        """
        Starting FL client core functions
        """
        self.register_client()        
        self.start_wait_model_server()
        self.start_model_exchange_server()

    def register_client(self):
        """
        Register an agent in aggregator
        """
        time.sleep(0.5)
        asyncio.get_event_loop().run_until_complete(self.participate())
    
    def start_wait_model_server(self):
        """
        Start a thread for waiting for global models
        """
        time.sleep(0.5)
        th = Thread(target = self._start_wait_model)
        th.start()

    def _start_wait_model(self):
        """
        Start a server for waiting for global models
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        receiving_serv = websockets.serve(self.wait_models, self.agent_ip, self.exch_socket, max_size=None, max_queue=None)
        loop.run_until_complete(asyncio.gather(receiving_serv))
        loop.run_forever()

    def start_model_exchange_server(self):
        """
        Start a thread for model exchange routine
        """
        time.sleep(0.5)
        self.agent_running = True
        th = Thread(target = self._start_model_exchange)
        th.start()

    def _start_model_exchange(self):
        """
        Start a loop for model exchange routine
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.gather(self.model_exchange_routine()))
        loop.run_forever()

    # Load and save models
    def load_model(self) -> Dict[str, Any]:
        """
        Read a global model file and return the models
        :return: Dict[str,np.array] - models
        """
        data_dict, _ = load_model_file(self.model_path, self.gmfile)
        return data_dict

    def load_global_model_data(self) -> List[Any]:
        """
        Read a global model file and return  ID and models
        :return: str - Model ID
        :return: np.array - models
        """
        data_dict, _ = load_model_file(self.model_path, self.gmfile)
        return data_dict['model_id'], data_dict['models']

    def save_model(self, model_id, models, meta_data_dict):
        """
        Save the trained models to the local file
        :param model_id: str - model ID
        :param models: np.array - models
        :param meta_data_dict: Dict with meta data
        :return:
        """
        data_dict = dict()
        data_dict['models'] = models
        gene_time = time.time()
        data_dict['model_id'] = model_id 
        data_dict['my_id'] = self.id
        data_dict['gene_time'] = gene_time

        save_model_file(data_dict, self.model_path, self.lmfile, meta_data_dict)


    # Read and change the client state
    def read_state(self) -> ClientState:
        """
        Read the value in the state file specified by model path
        :return: ClientState - A state indicated in the file
        """
        return read_state(self.model_path, self.statefile)

    def tran_state(self, state: ClientState):
        """
        Change the state of the agent
        State is indicated in two places: (1) local file 'state' and (2) waiting_flag
        :param state: ClientState
        :return:
        """
        self.waiting_flag = state
        write_state(self.model_path, self.statefile, self.waiting_flag)


    # Sending models
    def send_models(self, models, num_samples, perf_val):
        """
        Save the trained models to the local file
        :param models: np.array - models
        :param num_samples: int - Number of sample data
        :param perf_val: float - Performance data: accuracy in this case
        :return:
        """
        # Create a model ID
        model_id = generate_model_id(IDPrefix.agent, self.id, time.time())

        # Local Model evaluation (id, accuracy)
        meta_data_dict = dict()
        meta_data_dict["accuracy"] = perf_val
        meta_data_dict["num_samples"] = num_samples

        self.save_model(model_id, models, meta_data_dict)
        self.tran_state(ClientState.sending)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cl = Client()
    logging.info(f'--- Your IP is {cl.agent_ip} ---')

    cl.start_fl_client()

import asyncio
import time
import logging
import sys
import os
from typing import Dict, Any
from threading import Thread

from fl_main.lib.util.communication_handler import init_client_server, send, receive
from fl_main.lib.util.helpers import read_config, init_loop, \
     save_model_file, load_model_file, read_state, write_state, generate_id, \
     set_config_file, get_ip, compatible_data_dict_read, generate_model_id, \
     create_data_dict_from_models, create_meta_data_dict
from fl_main.lib.util.states import IDPrefix, ClientState, AggMsgType, ParticipateConfirmationMSGLocation, GMDistributionMsgLocation, PollingMSGLocation
from fl_main.lib.util.messengers import generate_lmodel_update_message, generate_agent_participation_message, generate_polling_message

class Client:
    """
    Client class instance provides the communication interface
    between Agent's ML logic and an aggregator
    """

    def __init__(self):

        time.sleep(2)
        logging.info(f"--- Agent initialized ---")

        self.agent_name = 'default_agent'

        # Unique ID in the system
        self.id = generate_id()

        # Getting IP Address of the agent itself
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
        self.reg_socket = self.config['reg_socket']
        self.msend_socket = 0  # later updated based on welcome message
        self.exch_socket = 0 

        if self.simulation_flag:
            # if it's simulation, use the manual socket number and agent name
            self.exch_socket = int(sys.argv[2])
            self.agent_name = sys.argv[3]

        # Local file location        
        self.model_path = f'{self.config["model_path"]}/{self.agent_name}'

        # if there is no directory to save models
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.lmfile = self.config['local_model_file_name']
        self.gmfile = self.config['global_model_file_name']
        self.statefile = self.config['state_file_name']

        # Aggregation round - later updated by the info from the aggregator
        self.round = 0
        
        # Initialization
        self.init_weights_flag = bool(self.config['init_weights_flag'])

        # Polling Method
        self.is_polling = bool(self.config['polling'])

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

        msg = generate_agent_participation_message(
                self.agent_name, self.id, model_id, models, self.init_weights_flag, self.simulation_flag,
                self.exch_socket, gene_time, performance_dict, self.agent_ip)
        resp = await send(msg, self.aggr_ip, self.reg_socket)

        logging.debug(msg)
        logging.info(f"--- Init Response: {resp} ---")

        # Parse the response message
        # including some socket info and the actual round number
        self.round = resp[int(ParticipateConfirmationMSGLocation.round)]
        self.exch_socket = resp[int(ParticipateConfirmationMSGLocation.exch_socket)]
        self.msend_socket = resp[int(ParticipateConfirmationMSGLocation.recv_socket)]
        self.id = resp[int(ParticipateConfirmationMSGLocation.agent_id)]

        # Receiving the welcome message
        logging.info(f'--- {resp[int(ParticipateConfirmationMSGLocation.msg_type)]} Message Received ---')

        self.save_model_from_message(resp, ParticipateConfirmationMSGLocation)

    async def model_exchange_routine(self):
        """
        Check the progress of training and send the updated models
        once the training is done
        :return:
        """
        while True:
            # Periodically check the state
            await asyncio.sleep(5)
            state = read_state(self.model_path, self.statefile)

            if state == ClientState.sending: 
                # Ready to send the local model
                await self.send_models()

            elif state == ClientState.waiting_gm:
                # Waiting for global models
                if self.is_polling == True:
                    await self.process_polling()
                else:
                    # Do nothing
                    logging.info(f'--- Waiting for Global Model ---')

            elif state == ClientState.training:
                # Local model is being trained, do nothing
                logging.info(f'--- Training is happening ---')

            elif state == ClientState.gm_ready:
                # Global model has been received, do nothing
                logging.info(f'--- Global Model is ready ---')

            else:
                logging.error(f'--- State Not Defined ---')
    

    # Push or Polling
    async def wait_models(self, websocket, path):
        """
        Waiting for cluster models from the aggregator
        :param websocket:
        :return:
        """
        gm_msg = await receive(websocket)
        logging.info(f'--- Global Model Received ---')

        logging.debug(f'Models: {gm_msg}')

        self.save_model_from_message(gm_msg, GMDistributionMsgLocation)
    
    async def process_polling(self):
        logging.info(f'--- Polling to see if there is any update ---')

        msg = generate_polling_message(self.round, self.id)
        resp = await send(msg, self.aggr_ip, self.msend_socket)
        if resp[int(PollingMSGLocation.msg_type)] == AggMsgType.update:
            logging.info(f'--- Global Model Received ---')
            self.save_model_from_message(resp, GMDistributionMsgLocation)
        else: # AggMsgType is "ack"
            logging.info(f'--- Global Model is NOT ready (ACK) ---')


    # Starting FL client functions
    def start_fl_client(self):
        """
        Starting FL client core functions
        """
        self.register_client()
        if self.is_polling == False:
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
        th = Thread(target = init_client_server, args=[self.wait_models, self.agent_ip, self.exch_socket])
        th.start()

    def start_model_exchange_server(self):
        """
        Start a thread for model exchange routine
        """
        time.sleep(0.5)
        self.agent_running = True
        th = Thread(target = init_loop, args=[self.model_exchange_routine()])
        th.start()

    # Save models from message
    def save_model_from_message(self, msg, MSG_LOC):

        # pass (model_id, models) to an app
        data_dict = create_data_dict_from_models(msg[int(MSG_LOC.model_id)], 
                        msg[int(MSG_LOC.global_models)], msg[int(MSG_LOC.aggregator_id)])
        self.round = msg[int(MSG_LOC.round)]

        # Save the received cluster global models to the local file
        save_model_file(data_dict, self.model_path, self.gmfile)
        logging.info(f'--- Global Models Saved ---')
        
        # State transition to gm_ready
        self.tran_state(ClientState.gm_ready)
        logging.info(f'--- Client State is now gm_ready ---')
    

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
        State is indicated in local file 'state'
        :param state: ClientState
        :return:
        """
        write_state(self.model_path, self.statefile, state)

    # Sending models
    async def send_models(self):
        # Read the models from the local file
        data_dict, performance_dict = load_model_file(self.model_path, self.lmfile)
        _, _, models, model_id = compatible_data_dict_read(data_dict)
        msg = generate_lmodel_update_message(self.id, model_id, models, performance_dict)

        logging.debug(f'Trained Models: {msg}')

        await send(msg, self.aggr_ip, self.msend_socket)
        logging.info('--- Local Models Sent ---')

        # State transition to waiting_gm
        self.tran_state(ClientState.waiting_gm)
        logging.info(f'--- Client State is now waiting_gm ---')

    def send_initial_model(self, initial_models, num_samples=1, perf_val=0.0):
        self.setup_sending_models(initial_models, num_samples, perf_val)

    def send_trained_model(self, models, num_samples, perf_value):
        # Check the state in case another global models arrived during the training
        state = self.read_state()
        if state == ClientState.gm_ready:
            # Do nothing: Discard the trained local models and adopt the new global models
            logging.info(f'--- The training was too slow. A new set of global models are available. ---')
        else:  # Keep the training results
            # Send models
            self.setup_sending_models(models, num_samples, perf_value)

    def setup_sending_models(self, models, num_samples, perf_val):
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
        meta_data_dict = create_meta_data_dict(perf_val, num_samples)
        data_dict = create_data_dict_from_models(model_id, models, self.id)
        save_model_file(data_dict, self.model_path, self.lmfile, meta_data_dict)
        logging.info(f'--- Local (Initial/Trained) Models saved ---')

        self.tran_state(ClientState.sending)
        logging.info(f'--- Client State is now sending ---')

    # Waiting models
    def wait_for_global_model(self):

        # Wait for global models (base models)
        while (self.read_state() != ClientState.gm_ready):
            time.sleep(5)

        # load models from the local file
        data_dict, _ = load_model_file(self.model_path, self.gmfile)
        global_models = data_dict['models']
        logging.info(f'--- Global Models read by Agent ---')

        self.tran_state(ClientState.training)
        logging.info(f'--- Client State is now training ---')

        return global_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cl = Client()
    logging.info(f'--- Your IP is {cl.agent_ip} ---')

    cl.start_fl_client()

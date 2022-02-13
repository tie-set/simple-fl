import pickle
import websockets
import asyncio
import logging
import time
import os
import pathlib
import sys
from typing import Any, List

from .sqlite_db import SQLiteDBHandler
from fl_main.lib.util.helpers import generate_id, read_config, set_config_file
from fl_main.lib.util.states import DBMsgType, DBPushMsgLocation, ModelType

class PseudoDB:
    """
    Pseudo Database class instance that receives models and their data from an aggregator,
    and pushes them to an actual database
    """

    def __init__(self):

        self.id = generate_id()

        self.module_path = pathlib.Path.cwd()

        self.simulation_flag = False
        if len(sys.argv) > 1:
            # if sys.argv[1] == '1', it's in simulation mode
            self.simulation_flag = bool(int(sys.argv[1]))

        # read the config file
        config_file = set_config_file(sys.argv, 2, "db")
        self.config = read_config(config_file)

        self.db_socket = self.config['db_socket']
        self.db_ip = self.config['db_ip']

        # Init DB
        self.data_path = self.config['db_data_path']

        # if there is no directory to save models
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.db_file = f'{self.data_path}/model_data{time.time()}.db'
        self.dbhandler = SQLiteDBHandler(self.db_file)
        self.dbhandler.initialize_DB()

        # Model save location
        # if there is no directory to save models
        self.db_model_path = self.config['db_model_path']
        if not os.path.exists(self.db_model_path):
            os.makedirs(self.db_model_path)


    async def handler(self, websocket, path):
        """
        Receives all requests from aggregators and returns requested info
        :param websocket:
        :param path:
        :return:
        """
        # receive a request from an aggregator
        bmsg = await websocket.recv()
        msg = pickle.loads(bmsg)
        logging.info(f'Request Arrived')
        logging.debug(f'Request: {msg}')

        # Extract the message type
        msg_type = msg[DBPushMsgLocation.msg_type]

        reply = list()
        if msg_type == DBMsgType.push: # models
            logging.info(f'--- Model pushed: {msg[int(DBPushMsgLocation.model_type)]} ---')
            self._push_all_data_to_db(msg)
            reply.append('confirmation')
        else:
            # Error for undefined message type
            raise TypeError(f'Undefined DB Access Message Type: {msg_type}.')

        # reply to the aggregator
        await websocket.send(pickle.dumps(reply))

    def _push_all_data_to_db(self, msg: List[Any]):
        pm = self._parse_message(msg)
        self.dbhandler.insert_an_entry(*pm)

        # save models
        model_id = msg[int(DBPushMsgLocation.model_id)]
        models = msg[int(DBPushMsgLocation.models)]
        fname = f'{self.db_model_path}/{model_id}.binaryfile'
        with open(fname, 'wb') as f:
            pickle.dump(models, f)

    def _parse_message(self, msg: List[Any]):
        component_id = msg[int(DBPushMsgLocation.component_id)]
        r = msg[int(DBPushMsgLocation.round)]
        mt = msg[int(DBPushMsgLocation.model_type)]
        model_id = msg[int(DBPushMsgLocation.model_id)]
        gene_time = msg[int(DBPushMsgLocation.gene_time)]
        meta_data = msg[int(DBPushMsgLocation.meta_data)]

        # if local model performance is saved
        local_prfmc = 0.0
        if mt == ModelType.local:
            try: local_prfmc = meta_data["accuracy"]
            except: pass

        # Number of samples is saved
        num_samples = 0
        try: num_samples = meta_data["num_samples"]
        except: pass

        return component_id, r, mt, model_id, gene_time, local_prfmc, num_samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pdb = PseudoDB()
    start_server = websockets.serve(pdb.handler, pdb.db_ip, pdb.db_socket,
                                    max_size=None, max_queue=None)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()
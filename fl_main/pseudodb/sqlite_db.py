import sqlite3
import datetime
import logging

# Message type between aggregators and DB
from fl_main.lib.util.states import ModelType

class SQLiteDBHandler:
    """
        SQLiteDB Handler class that creates and initialize SQLite DB,
        and inserts models to the SQLiteDB
    """

    def __init__(self, db_file):
        self.db_file = db_file

    def initialize_DB(self):
        conn = sqlite3.connect(f'{self.db_file}')
        c = conn.cursor()

        # create table for each model types
        # local
        c.execute('''CREATE TABLE local_models(model_id, generation_time, agent_id, round, performance, num_samples)''')

        # cluster
        c.execute('''CREATE TABLE cluster_models(model_id, generation_time, aggregator_id, round, num_samples)''')

        conn.commit()
        conn.close()

    def insert_an_entry(self,
                         component_id: str,
                         r: int,
                         mt: ModelType,
                         model_id: str,
                         gtime: float,
                         local_prfmc: float,
                         num_samples: int
                        ):

        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        t = datetime.datetime.fromtimestamp(gtime)
        gene_time = t.strftime('%m/%d/%Y %H:%M:%S')

        if mt == ModelType.local:
            c.execute('''INSERT INTO local_models VALUES (?, ?, ?, ?, ?, ?);''', (model_id, gene_time, component_id, r, local_prfmc, num_samples))
            logging.info(f"--- Local Models are saved ---")

        elif mt == ModelType.cluster:
            c.execute('''INSERT INTO cluster_models VALUES (?, ?, ?, ?, ?);''', (model_id, gene_time, component_id, r, num_samples))
            logging.info(f"--- Cluster Models are saved ---")

        else:
            logging.info(f"--- Nothing saved ---")

        conn.commit()
        conn.close()

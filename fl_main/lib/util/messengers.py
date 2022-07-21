import time
import numpy as np
from typing import Dict, List, Any
from fl_main.lib.util.states import ModelType, DBMsgType, AgentMsgType, AggMsgType

def generate_db_push_message(component_id: str,
                             round: int,
                             model_type: ModelType,
                             models: Dict[str,np.array],
                             model_id: str,
                             gene_time: float,
                             performance_dict: Dict[str,float]) -> List[Any]:
    msg = list()
    msg.append(DBMsgType.push)  # 0
    msg.append(component_id)  # 1
    msg.append(round)  # 2
    msg.append(model_type)  # 3
    msg.append(models)  # 4
    msg.append(model_id)  # 5
    msg.append(gene_time)  # 6
    msg.append(performance_dict)  # 7
    return msg

def generate_lmodel_update_message(agent_id: str,
                                   model_id: str,
                                   local_models: Dict[str,np.array],
                                   performance_dict: Dict[str,float]) -> List[Any]:
    msg = list()
    msg.append(AgentMsgType.update)  # 0
    msg.append(agent_id)  # 1
    msg.append(model_id)  # 2
    msg.append(local_models)  # 3
    msg.append(time.time())  # 4
    msg.append(performance_dict)  # 5
    return msg

def generate_cluster_model_dist_message(aggregator_id: str,
                                        model_id: str,
                                        round: int,
                                        models: Dict[str,np.array]) -> List[Any]:
    msg = list()
    msg.append(AggMsgType.update)  # 0
    msg.append(aggregator_id)  # 1
    msg.append(model_id)  # 2
    msg.append(round)  # 3
    msg.append(models)  # 4
    return msg

def generate_agent_participation_message(agent_name: str,
                                         agent_id: str,
                                         model_id: str,
                                         models: Dict[str,np.array],
                                         init_weights_flag: bool,
                                         simulation_flag: bool,
                                         exch_socket: str,
                                         gene_time: float,
                                         meta_dict: Dict[str,float],
                                         agent_ip: str) -> List[Any]:
    msg = list()
    msg.append(AgentMsgType.participate)  # 0
    msg.append(agent_id)  # 1
    msg.append(model_id)  # 2
    msg.append(models)  # 3
    msg.append(init_weights_flag)  # 4
    msg.append(simulation_flag)  # 5
    msg.append(exch_socket)  # 6
    msg.append(gene_time)  # 7
    msg.append(meta_dict)  # 8
    msg.append(agent_ip)  # 9
    msg.append(agent_name)  # 9
    return msg

def generate_agent_participation_confirm_message(aggregator_id: str,
                                                      model_id: str,
                                                      models: Dict[str,np.array],
                                                      round: int,
                                                      agent_id: str,
                                                      exch_socket: str,
                                                      recv_socket: str) -> List[Any]:
    msg = list()
    msg.append(AggMsgType.welcome)  # 0
    msg.append(aggregator_id)  # 1
    msg.append(model_id)  # 2
    msg.append(models)  # 3    
    msg.append(round)  # 4
    msg.append(agent_id) # 5
    msg.append(exch_socket)  # 6
    msg.append(recv_socket)  # 7
    return msg

def generate_ack_message():
    msg = list()
    msg.append(AggMsgType.ack) # 0
    return msg

def generate_polling_message(round: int, agent_id: str):
    msg = list()
    msg.append(AgentMsgType.polling) # 0
    msg.append(round) # 1
    msg.append(agent_id) # 2
    return msg


# ```setups```
This directory stores configuration files for (1) environment set-up and (2) aggregator/agent/DB/model set-up 

## ```federatedenv.yaml```
This yaml file contains all necessary python packages to run this simple FL framework.
You can create a virtual Anaconda environment with this script:
```sh
conda env create -n federatedenv -f ./setups/federatedenv.yaml
```

Note: The environment has ```Python 3.7.4```. There is some known issues of ```ipfshttpclient``` with ```Python 3.7.2 and older```.


## ```config json files```
These json files are read by aggregators, agents, and DB handlers to configure their initial setups.

### ```config_db.json```
- ```db_ip```: An DB IP address
    - e.g. ```localhost```
- ```db_socket```: A socket number used between DB and an aggregator.
    - e.g. ```9017```
- ```db_name```: Name of the SQLite database.
    - e.g. ```sample_data```
- ```db_data_path```: Path to the SQLite database.
    - e.g. ```./db```
- ```db_model_path```: Path to the directory to save all ML models
    - e.g. ```./db/models```
  
### ```config_aggregator.json```
- ```aggr_ip```: An aggregator IP address
    - e.g. ```localhost``` 
- ```db_ip```: An DB IP address
    - e.g. ```localhost```
- ```reg_socket```: A socket number used by agents to join an aggregator for the first time.
    - e.g. ```8765```
- ```exch_socket```: A socket number used to upload local models to an aggregator from an agent. Agents will get to know this socket from the communications with an aggregator.
    - e.g. ```7890```
- ```recv_socket```: A socket number used to send back semi global models to an agent from an aggregator. Agents will get to know this socket from the communications with an aggregator.
    - e.g. ```4321```
- ```db_socket```: A socket number used between DB and an aggregator.
    - e.g. ```9017```
- ```round_interval```: Period of time after which an agent check if there are enough number of models to start an aggregation step. (Unit: seconds)
  - e.g. ```5```
- ```aggregation_threshold```: Percentage of the number of collected local models required to start an aggregation step
    - e.g. ```1.0```, ```0.8```
- ```polling```: A flag for using a polling method or not. If 1, use the polling method, otherwise use a push method.
    - e.g. ```1```

### ```config_agent.json```
- ```aggr_ip```: An aggregator IP address
    - e.g. ```localhost```  
- ```reg_socket```: A socket number used by agents to join an aggregator for the first time.
    - e.g. ```8765```
- ```model_path```: A path to a local director in the agent machine to save local models and some state info. 
  - e.g. ```"./data/agents"```
- ```local_model_file_name```: A file name to save local models in the agent machine. 
  - e.g. ```lms.binaryfile```
- ```global_model_file_name```: A file name to save local models in the agent machine. 
  - e.g. ```gms.binaryfile```
- ```state_file_name```: A file name to store the agent state in the agent machine.
    - e.g. ```state```
- ```init_weights_flag```: 1 if the weights are initialized with certain values, 0 otherwise where weights are initialized with zeros.
    - e.g. ```1```
- ```polling```: A flag for using a polling method or not. If 1, use the polling method, otherwise use a push method.
    - e.g. ```1```
# ```Simple FL``` package

## ```agent```
Agent-side modules.

- ```client.py```: A ```Client``` class instance provides the communication interface between Agent's ML logic and an aggregator.


## ```aggregator```
Aggregator-side modules

## ```pseudodb```
Pseudo DB modules


## ```lib```
General function library including helper functions used across the entire codes.

### ```lib/util```
- ```data_struc.py```: Converting LimitedDict into Dict.
- ```helpers.py```: General helper functions used by agents, aggregators, and DB handlers.
- ```messengers.py```: Generating lit type of messages based on input parameters.
- ```states.py```: Message types and Agent states are defined.

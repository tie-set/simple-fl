import numpy as np
import time
import logging
import sys
from typing import Dict

from fl_main.agent.client import Client

def init_models() -> Dict[str,np.array]:
    """
    Return the templates of models (in a dict) to tell the structure
    The models need not to be trained
    :return: Dict[str,np.array]
    """
    models = dict()
    models['model1'] = np.array([[1, 2, 3], [4, 5, 6]])
    models['model2'] = np.array([[1, 2], [3, 4]])

    if len(sys.argv) > 4:
        if sys.argv[4] == 'a2':
            models['model1'] = np.array([[3, 4, 5], [6, 7, 8]])
            models['model2'] = np.array([[3, 4], [5, 6]])

    logging.info(f'--- Model template generated ---')
    return models

def training(models: Dict[str,np.array], init_flag: bool = False) -> Dict[str,np.array]:
    """
    A place holder function for each ML application
    Return the trained models
    Note that each models should be decomposed into numpy arrays
    Logic should be in the form: models -- training --> new local models
    :param models: Dict[str,np.array]
    :param init_flag: bool - True if it's at the init step.
    False if it's an actual training step
    :return: Dict[str,np.array] - trained models
    """
    # return templates of models to tell the structure
    # This model is not necessarily actually trained
    if init_flag:
        return init_models()

    # Do ML Training
    logging.info(f'--- Training ---')
    # In this example, there is no actual training
    # Please replace this part with your ML logic
    # models -- training --> new local models
    models = dict()
    models['model1'] = np.array([[1, 2, 3], [4, 5, 6]])
    models['model2'] = np.array([[1, 2], [3, 4]])

    if len(sys.argv) > 4:
        if sys.argv[4] == 'a2':
            models['model1'] = np.array([[3, 4, 5], [6, 7, 8]])
            models['model2'] = np.array([[3, 4], [5, 6]])

    time.sleep(5)

    print("models", models)

    return models

def compute_performance(models: Dict[str,np.array], testdata) -> float:
    """
    Given a set of models and test dataset, compute the performance of the models
    :param models:
    :param testdata:
    :return:
    """
    # replace this with actual performance computation logic
    accuracy = 0.5
    return accuracy

def judge_termination(training_count: int = 0, global_arrival_count: int = 0) -> bool:
    """
    Decide if it finishes training process and exits from FL platform
    :param training_count: int - the number of training done
    :param global_arrival_count: int - the number of times it received global models
    :return: bool - True if it continues the training loop; False if it stops
    """

    # Depending on the criteria for termination,
    # change the return bool value
    # could call a performance tracker to check if the current models satisfy the required performance
    return True

def prep_test_data():
    testdata = 0
    return testdata


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('--- This is a minimal example ---')

    cl = Client()
    logging.info(f'--- Your IP is {cl.agent_ip} ---')

    # Create a set of template models (to tell the shapes)
    initial_models = training(dict(), init_flag=True)

    # Sending initial models
    cl.send_models(initial_models, 1, 0.0)

    # Starting FL client 
    cl.start_fl_client()

    training_count = 0
    gm_arrival_count = 0
    while judge_termination(training_count, gm_arrival_count):
        # Check the state file saved locally
        state = cl.read_state()

        # Wait for global models (base models)
        if state == cl.ClientState.gm_ready:
            gm_arrival_count += 1
            logging.info(f'--- Reading Global models ---')

            # load models from the local file
            global_model_id, global_models = cl.load_global_model_data()
            print('global_models:', global_models)

            cl.tran_state(cl.ClientState.training)

            # Global Model evaluation (id, accuracy)
            global_model_performance_data = compute_performance(global_models, prep_test_data())

        # Training
        if state == cl.ClientState.training:
            models = training(global_models)
            training_count += 1
            logging.info(f'--- Training Done ---')

            # Check the state in case another global models arrived during the training
            state = cl.read_state()
            if state == cl.ClientState.gm_ready:
                # Do nothing
                # Discard the trained local models and adopt the new global models
                logging.info(f'--- The training was too slow. A new set of global models are available. ---')

            else:  # Keep the training results
                # Local Model evaluation (id, accuracy)
                performance_value = compute_performance(models, prep_test_data())

                # Send models
                cl.send_models(models, 1, performance_value)

                logging.info(f'--- Normal transition: The trained local models saved ---')

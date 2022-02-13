import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cnn import Net
from .conversion import Converter
from .ic_training import DataManger, execute_ic_training

from fl_main.agent.client import Client


class TrainingMetaData:
    # The number of training data used for each round
    # This will be used for the weighted averaging
    # Set to a natural number > 0
    num_training_data = 8000

def init_models() -> Dict[str,np.array]:
    """
    Return the templates of models (in a dict) to tell the structure
    The models need not to be trained
    :return: Dict[str,np.array]
    """
    net = Net()
    return Converter.cvtr().convert_nn_to_dict_nparray(net)

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
        # Prepare the training data
        # num of samples / 4 = threshold for training due to the batch size

        DataManger.dm(int(TrainingMetaData.num_training_data / 4))
        return init_models()

    # Do ML Training
    logging.info(f'--- Training ---')

    # Create a CNN based on global (cluster) models
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # models -- training --> new local models
    trained_net = execute_ic_training(DataManger.dm(), net, criterion, optimizer)
    models = Converter.cvtr().convert_nn_to_dict_nparray(trained_net)
    return models

def compute_performance(models: Dict[str,np.array], testdata, is_local: bool) -> float:
    """
    Given a set of models and test dataset, compute the performance of the models
    :param models:
    :param testdata:
    :return:
    """
    # Convert np arrays to a CNN
    net = Converter.cvtr().convert_dict_nparray_to_nn(models)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in DataManger.dm().testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = float(correct) / total

    mt = 'local'
    if not is_local:
        mt = 'Global'

    print(f'Accuracy of the {mt} model with the 10000 test images: {100 * acc} %%')

    return acc

def judge_termination(training_count: int = 0, gm_arrival_count: int = 0) -> bool:
    """
    Decide if it finishes training process and exits from FL platform
    :param training_count: int - the number of training done
    :param gm_arrival_count: int - the number of times it received global models
    :return: bool - True if it continues the training loop; False if it stops
    """

    # could call a performance tracker to check if the current models satisfy the required performance
    return True

def prep_test_data():
    testdata = 0
    return testdata

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('--- This is a demo of Image Classification with Federated Learning ---')

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

        # Wait for Global models (base models)
        if state == cl.ClientState.gm_ready:
            gm_arrival_count += 1
            logging.info(f'--- Reading Global models ---')

            # load models from the local file
            global_model_id, global_models = cl.load_global_model_data()

            cl.tran_state(cl.ClientState.training)

            # Global Model evaluation (id, accuracy)
            global_model_performance_data = compute_performance(global_models, prep_test_data(), False)

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
                # Send # of training data
                # meta_data_dict[cl.getReservedKeys().rkeys[0]] = int(TrainingMetaData.num_training_data)

                print("int(TrainingMetaData.num_training_data)", int(TrainingMetaData.num_training_data))
                accuracy = compute_performance(models, prep_test_data(), True)

                # Send models with accuracy data
                cl.send_models(models, int(TrainingMetaData.num_training_data), accuracy)

                logging.info(f'--- Normal transition: The trained local models saved ---')

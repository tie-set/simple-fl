import websockets
import asyncio
import pickle
import logging

def init_db_server(func, ip, socket):
    """
    Start the DB server
    """
    start_server = websockets.serve(func, ip, socket,
                                    max_size=None, max_queue=None)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

def init_fl_server(register, receive_local_models, model_synthesis_routine, aggr_ip, reg_socket, recv_socket):
    """
    Start the FL server
    """
    loop = asyncio.get_event_loop()
    start_server = websockets.serve(register, aggr_ip, reg_socket,
                                    max_size=None, max_queue=None)
    start_receiver = websockets.serve(receive_local_models, aggr_ip, recv_socket,
                                      max_size=None, max_queue=None)
    loop.run_until_complete(asyncio.gather(start_server,
                                           start_receiver,
                                           model_synthesis_routine))
    loop.run_forever()

def init_client_server(func, ip, socket):
    """
    Start the client server
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client_server = websockets.serve(func, ip, socket, max_size=None, max_queue=None)
    loop.run_until_complete(asyncio.gather(client_server))
    loop.run_forever()

async def send(msg, ip, socket):
    """
    Send a message to the ip and socket
    """
    resp = None
    try:
        wsaddr = f'ws://{ip}:{socket}'
        async with websockets.connect(wsaddr, max_size=None, max_queue=None, ping_interval=None) as websocket:
            await websocket.send(pickle.dumps(msg))

            try:
                rmsg = await websocket.recv()
                resp = pickle.loads(rmsg)
            except:
                logging.info("--- Nothing to be received ---")
                pass

            return resp
    except:
        logging.error("Connection lost to the agent: " + ip)
        logging.error(f'--- Message NOT Sent ---')

async def send_websocket(msg, websocket):
    """
    Send a binary file (message) to an agent through a give websocket
    :param bsgms: Message (binary file)
    :param websocket:
    :return:
    """
    while not websocket:  # wait until socket being initialized
        await asyncio.sleep(0.001)
    await websocket.send(pickle.dumps(msg))

async def receive(websocket):
    """
    Receive the message from the websocket
    """
    return pickle.loads(await websocket.recv())
    
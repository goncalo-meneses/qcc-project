from netqasm.sdk.classical_communication.message import StructuredMessage

def bbpssw_protocol_alice(q1, q2, alice, socket):
    """
    Implements Alice's side of the BBPSSW distillation protocol.
    This function should perform the gates and measurements for BBPSSW using
    qubits q1 and q2, then send the measurement outcome to Bob and determine
    if the distillation was successful.
    
    :param q1: Alice's qubit from the first entangled pair
    :param q2: Alice's qubit from the second entangled pair
    :param alice: Alice's NetQASMConnection
    :param socket: Alice's classical communication socket to Bob
    :return: True/False indicating if protocol was successful
    """
    a = bbpssw_gates_and_measurement_alice(q1, q2)
    alice.flush()

    ma = int(a)
    socket.send_structured(StructuredMessage("m_a: ", ma))

    mb = socket.recv_structured().payload

    success = False
    if ma == mb:
        success = True

    return success


def bbpssw_gates_and_measurement_alice(q1, q2):
    """
    Performs the gates and measurements for Alice's side of the BBPSSW protocol
    :param q1: Alice's qubit from the first entangled pair
    :param q2: Alice's qubit from the second entangled pair
    :return: Integer 0/1 indicating Alice's measurement outcome
    """
    q1.cnot(q2)
    a = q2.measure()

    return a

def bbpssw_protocol_bob(q1, q2, bob, socket):
    """
    Implements Bob's side of the BBPSSW distillation protocol.
    This function should perform the gates and measurements for BBPSSW using
    qubits q1 and q2, then send the measurement outcome to Alice and determine
    if the distillation was successful.
    
    :param q1: Bob's qubit from the first entangled pair
    :param q2: Bob's qubit from the second entangled pair
    :param bob: Bob's NetQASMConnection
    :param socket: Alice's classical communication socket to Bob
    :return: True/False indicating if protocol was successful
    """
    b = bbpssw_gates_and_measurement_bob(q1, q2)
    bob.flush()

    mb = int(b)
    socket.send_structured(StructuredMessage("m_b: ", mb))

    ma = socket.recv_structured().payload

    success = False
    if ma == mb:
        success = True

    return success

def bbpssw_gates_and_measurement_bob(q1, q2):
    """
    Performs the gates and measurements for Bob's side of the BBPSSW protocol
    :param q1: Bob's qubit from the first entangled pair
    :param q2: Bob's qubit from the second entangled pair
    :return: Integer 0/1 indicating Bob's measurement outcome
    """
    q1.cnot(q2)
    q2.H()
    b = q2.measure()

    return b

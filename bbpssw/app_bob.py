from bbpssw import bbpssw_protocol_bob
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state

def main(app_config=None):

    socket = Socket("bob", "alice")

    epr_socket = EPRSocket("alice")

    bob = NetQASMConnection(
        app_name=app_config.app_name,
        epr_sockets=[epr_socket]
    )

    with bob:
        epr1, epr2 = epr_socket.recv(number=2)

        bob_method = bbpssw_protocol_bob(epr1, epr2, bob, socket)

    print(bob_method)

if __name__ == "__main__":
    main()

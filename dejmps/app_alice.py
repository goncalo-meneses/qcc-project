from dejmps import dejmps_protocol_alice
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket

def main(app_config=None):

    socket = Socket("alice", "bob", log_config=app_config.log_config)

    epr_socket = EPRSocket("bob")

    alice = NetQASMConnection("alice", log_config=app_config.log_config, epr_sockets=[epr_socket])

    with alice:
        epr_1, epr_2 = epr_socket.create(number=2)

        succ = dejmps_protocol_alice(epr_1, epr_2, alice, socket)

if __name__ == "__main__":
    main()

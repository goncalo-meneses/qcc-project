from bbpssw import bbpssw_protocol_alice
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket

def main(app_config=None):

    socket = Socket("alice", "bob")

    epr_socket = EPRSocket("bob")

    alice = NetQASMConnection(
        app_name = app_config.app_name,
        epr_sockets = [epr_socket]
    )

    with alice:
        epr1, epr2 = epr_socket.create(number=2)

        alice_method = bbpssw_protocol_alice(epr1, epr2, alice, socket)

    print(alice_method)

if __name__ == "__main__":
    main()

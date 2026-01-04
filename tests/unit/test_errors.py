import socket

def connect_to_server(host: str, port: int):
    """Attempt to connect to a TCP server that doesn't exist."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)  # 5 second timeout

    try:
        sock.connect((host, port))
        print(f"Connected to {host}:{port}")
    except ConnectionRefusedError as e:
        print(f"Connection refused: {e}")
        raise
    except socket.timeout as e:
        print(f"Connection timed out: {e}")
        raise
    finally:
        sock.close()

def send_udp(host: str, port: int, message: bytes):
    """Send UDP datagram."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)

    try:
        # sendto() doesn't establish connection, just sends
        sock.sendto(message, (host, port))
        print(f"Sent UDP packet to {host}:{port}")

        # recvfrom() can timeout waiting for response
        data, addr = sock.recvfrom(1024)
        print(f"Received: {data}")
    except socket.gaierror as e:
        # DNS resolution failure (same as TCP)
        print(f"Name resolution error: {e}")
        raise
    except socket.timeout as e:
        # No response received within timeout
        print(f"Timeout waiting for response: {e}")
        raise
    except OSError as e:
        # Can occur if network is unreachable, port unreachable (ICMP)
        print(f"Network error: {e}")
        raise
    finally:
        sock.close()


class TestErrors:

    def test_connection_error(self):
        try:
            connect_to_server("localhost", 9999)

            assert False, "Expected ConnectionRefusedError was not raised."
        except ConnectionError as e:
            print(f"Caught ConnectionError: {type(e).__name__}")

    def test_get_addr_error(self):
        try:
            connect_to_server("lokalhost", 9999)

            assert False, "Expected socket.gaierror was not raised."
        except socket.gaierror as e:
            print(f"Name resolution error: {e}")

    def test_udp_timeout(self):
        try:
            send_udp("localhost", 9999, b"test")

            assert False, "Expected timeout"
        except socket.timeout as e:
            print(f"UDP timeout: {e}")


def test_tcp_timeout():
    """Connect to a server that accepts connections but never sends data."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)

    try:
        # Connect to a server that accepts but doesn't respond
        # (e.g., nc -l 9999 on Linux)
        sock.connect(("localhost", 9999))

        # This will timeout because server doesn't send data
        data = sock.recv(4096)  # Raises socket.timeout after 2 seconds

    except socket.timeout as e:
        print(f"Timeout during recv: {e}")
    except ConnectionRefusedError as e:
        print(f"Connection refused: {e}")
    finally:
        sock.close()


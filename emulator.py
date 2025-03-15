import argparse
import socket
import threading
import subprocess
import time
import sys

# Predefined neural network configurations:
# Weight size in bytes and GPU computation time in seconds.
NN_MODELS = {
    "ResNet50": {"weight_size": 100 * 1024 * 1024, "gpu_time": 2.0},
    "VGG16":    {"weight_size": 250 * 1024 * 1024, "gpu_time": 3.0},
    "BERT":     {"weight_size": 420 * 1024 * 1024, "gpu_time": 4.0}
}

# Base ports for iperf sessions (you can adjust these)
UPDATE_BASE_PORT = 6000      # For worker-to-PS updates.
BROADCAST_BASE_PORT = 7000   # For PS-to-worker broadcasts.

#########################################
# Parameter Server (PS) Implementation
#########################################

class ParameterServer:
    def __init__(self, control_port, num_workers, num_iterations, weight_size):
        self.control_port = control_port
        self.num_workers = num_workers
        self.num_iterations = num_iterations
        self.weight_size = weight_size

        # This dict will map worker_id to a tuple (control_socket, worker_addr)
        self.worker_controls = {}
        self.lock = threading.Lock()

    def start_control_server(self):
        """Start a control server to accept connections from workers."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("", self.control_port))
        server_sock.listen(self.num_workers)
        print(f"PS: Control server listening on port {self.control_port}. Waiting for {self.num_workers} workers...")
        while len(self.worker_controls) < self.num_workers:
            conn, addr = server_sock.accept()
            # Expect the worker to send its worker_id as the first message.
            worker_id = conn.recv(1024).decode().strip()
            if worker_id:
                with self.lock:
                    self.worker_controls[worker_id] = (conn, addr)
                print(f"PS: Registered worker {worker_id} from {addr}.")
            else:
                conn.close()
        print("PS: All workers have connected.")
        self.control_server_sock = server_sock

    def run(self):
        """Run training iterations with update and broadcast phases using iperf."""
        for iteration in range(self.num_iterations):
            print(f"\nPS: Starting iteration {iteration}.")
            # --- Update Phase: Workers send update data to PS ---
            update_threads = []
            with self.lock:
                for worker_id, (ctrl_sock, worker_addr) in self.worker_controls.items():
                    port = UPDATE_BASE_PORT + int(worker_id)
                    t = threading.Thread(target=self.handle_update_phase,
                                         args=(worker_id, ctrl_sock, worker_addr, iteration, port))
                    t.start()
                    update_threads.append(t)
            # Wait for all update sessions to finish.
            for t in update_threads:
                t.join()

            # --- Broadcast Phase: PS sends weights to workers ---
            broadcast_threads = []
            with self.lock:
                for worker_id, (ctrl_sock, worker_addr) in self.worker_controls.items():
                    port = BROADCAST_BASE_PORT + int(worker_id)
                    t = threading.Thread(target=self.handle_broadcast_phase,
                                         args=(worker_id, ctrl_sock, worker_addr, iteration, port))
                    t.start()
                    broadcast_threads.append(t)
            for t in broadcast_threads:
                t.join()

            print(f"PS: Iteration {iteration} completed.\n")
            # Small pause between iterations.
            time.sleep(0.2)
        print("PS: All iterations completed. Shutting down.")
        # Close control sockets.
        with self.lock:
            for worker_id, (conn, _) in self.worker_controls.items():
                conn.close()
        self.control_server_sock.close()

    def handle_update_phase(self, worker_id, ctrl_sock, worker_addr, iteration, port):
        """
        In update phase, instruct the worker to send its update via iperf.
        PS starts an iperf server (with --one-off) on a dedicated port.
        """
        # Send control message: "update <iteration> <port>"
        message = f"update {iteration} {port}\n"
        try:
            ctrl_sock.sendall(message.encode())
        except Exception as e:
            print(f"PS: Failed to send update command to worker {worker_id}: {e}")
            return

        print(f"PS: [Iteration {iteration}] Starting iperf server for update from worker {worker_id} on port {port}.")
        # Start iperf3 server to receive data.
        cmd = ["iperf3", "-s", "--one-off", "-p", str(port)]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            print(f"PS: [Iteration {iteration}] Update session from worker {worker_id} completed.")
        except subprocess.TimeoutExpired:
            print(f"PS: [Iteration {iteration}] Update session from worker {worker_id} timed out.")

    def handle_broadcast_phase(self, worker_id, ctrl_sock, worker_addr, iteration, port):
        """
        In broadcast phase, instruct the worker to start an iperf server to receive data.
        Then PS acts as iperf client and sends the weights.
        """
        # Send control message: "broadcast <iteration> <port>"
        message = f"broadcast {iteration} {port}\n"
        try:
            ctrl_sock.sendall(message.encode())
        except Exception as e:
            print(f"PS: Failed to send broadcast command to worker {worker_id}: {e}")
            return

        print(f"PS: [Iteration {iteration}] Starting iperf client for broadcast to worker {worker_id} on port {port}.")
        # Use the worker's IP (from control connection) as the target.
        worker_ip = worker_addr[0]
        cmd = ["iperf3", "-c", worker_ip, "-p", str(port), "-n", str(self.weight_size)]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            print(f"PS: [Iteration {iteration}] Broadcast to worker {worker_id} completed.")
        except subprocess.TimeoutExpired:
            print(f"PS: [Iteration {iteration}] Broadcast to worker {worker_id} timed out.")

#########################################
# Worker Implementation
#########################################

class Worker:
    def __init__(self, ps_host, control_port, worker_id, num_iterations, weight_size, gpu_time):
        self.ps_host = ps_host
        self.control_port = control_port
        self.worker_id = worker_id
        self.num_iterations = num_iterations
        self.weight_size = weight_size
        self.gpu_time = gpu_time
        self.ctrl_sock = None
        self.ctrl_file = None  # File-like object for reading control messages

    def connect_control(self):
        """Connect to the PS control channel and send our worker_id."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.ps_host, self.control_port))
            # Wrap the socket for line-based reading.
            self.ctrl_file = sock.makefile('r')
            # Send worker_id as the first message.
            sock.sendall(f"{self.worker_id}\n".encode())
            self.ctrl_sock = sock
            print(f"Worker {self.worker_id}: Connected to PS control at {self.ps_host}:{self.control_port}.")
        except Exception as e:
            print(f"Worker {self.worker_id}: Control connection failed: {e}")
            sys.exit(1)

    def wait_for_control_command(self, cmd_type, expected_iteration):
        """
        Wait for a control command from the PS.
        Expected commands:
          - "update <iteration> <port>"
          - "broadcast <iteration> <port>"
        This version uses the file-like object to reliably read one line at a time.
        """
        try:
            line = self.ctrl_file.readline()
            if not line:
                print(f"Worker {self.worker_id}: Control connection closed unexpectedly.")
                sys.exit(1)
            line = line.strip()
            parts = line.split()
            if len(parts) != 3 or parts[0] != cmd_type or int(parts[1]) != expected_iteration:
                print(f"Worker {self.worker_id}: Unexpected control message: {line}")
                return None
            return parts
        except Exception as e:
            print(f"Worker {self.worker_id}: Error reading control command: {e}")
            return None

    def run(self):
        self.connect_control()
        for iteration in range(self.num_iterations):
            print(f"\nWorker {self.worker_id}: Starting iteration {iteration}.")
            # Simulate GPU computation.
            print(f"Worker {self.worker_id}: Simulating GPU computation for {self.gpu_time:.2f} sec.")
            time.sleep(self.gpu_time)

            # --- Update Phase: Send update to PS ---
            update_cmd = self.wait_for_control_command("update", iteration)
            if update_cmd:
                port = int(update_cmd[2])
                print(f"Worker {self.worker_id}: Received update command for iteration {iteration} on port {port}.")
                # Start iperf client to send data.
                cmd = ["iperf3", "-c", self.ps_host, "-p", str(port), "-n", str(self.weight_size)]
                try:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                    print(f"Worker {self.worker_id}: Update iperf session completed for iteration {iteration}.")
                except subprocess.TimeoutExpired:
                    print(f"Worker {self.worker_id}: Update iperf session timed out for iteration {iteration}.")

            # --- Broadcast Phase: Receive weights from PS ---
            broadcast_cmd = self.wait_for_control_command("broadcast", iteration)
            if broadcast_cmd:
                port = int(broadcast_cmd[2])
                print(f"Worker {self.worker_id}: Received broadcast command for iteration {iteration} on port {port}.")
                # Start iperf server to receive data.
                cmd = ["iperf3", "-s", "--one-off", "-p", str(port)]
                try:
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                    print(f"Worker {self.worker_id}: Broadcast iperf session completed for iteration {iteration}.")
                except subprocess.TimeoutExpired:
                    print(f"Worker {self.worker_id}: Broadcast iperf session timed out for iteration {iteration}.")

            # Small pause before next iteration.
            time.sleep(0.2)
        print(f"Worker {self.worker_id}: All iterations completed. Closing control connection.")
        self.ctrl_file.close()
        self.ctrl_sock.close()

#########################################
# Main: Argument Parsing and Mode Selection
#########################################

def main():
    parser = argparse.ArgumentParser(description="Distributed Training Emulator using iperf3 for real data transfer")
    parser.add_argument("--mode", choices=["ps", "worker"], required=True,
                        help="Run as parameter server (ps) or worker")
    parser.add_argument("--port", type=int, default=5000,
                        help="Control channel port for PS to listen on / workers to connect to")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="(PS mode) Number of workers expected to connect")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of training iterations")
    parser.add_argument("--ps_host", type=str, default="localhost",
                        help="(Worker mode) PS control host IP address to connect to")
    parser.add_argument("--worker_id", type=str, default="0",
                        help="(Worker mode) Identifier for this worker")
    parser.add_argument("--nn", choices=list(NN_MODELS.keys()), default="ResNet50",
                        help="Neural network model to emulate")
    args = parser.parse_args()

    nn_config = NN_MODELS.get(args.nn)
    weight_size = nn_config["weight_size"]
    gpu_time = nn_config["gpu_time"]

    if args.mode == "ps":
        ps = ParameterServer(control_port=args.port,
                             num_workers=args.num_workers,
                             num_iterations=args.num_iterations,
                             weight_size=weight_size)
        ps.start_control_server()
        ps.run()
    else:
        worker = Worker(ps_host=args.ps_host,
                        control_port=args.port,
                        worker_id=args.worker_id,
                        num_iterations=args.num_iterations,
                        weight_size=weight_size,
                        gpu_time=gpu_time)
        worker.run()

if __name__ == "__main__":
    main()

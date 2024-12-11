import os
import subprocess
import multiprocessing
import time
import signal
import sys


def run_ollama_instance(port, model=None):
    """
    Run an Ollama instance on a specific port

    Args:
        port (int): Port number for the Ollama instance
        model (str, optional): Specific model to load for this instance
    """
    env = os.environ.copy()
    env['OLLAMA_HOST'] = f'127.0.0.1:{port}'

    # Prepare the command
    command = ['ollama', 'serve']

    # Optional: If a specific model is provided
    if model:
        command.extend(['--model', model])

    try:
        print(f"Starting Ollama instance on port {port}")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Log output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[Port {port}] {output.strip()}")

        # Check for errors
        stderr = process.stderr.read()
        if stderr:
            print(f"Error on port {port}: {stderr}")

        return process.returncode

    except Exception as e:
        print(f"Exception on port {port}: {e}")
        return 1


def run_multiple_ollama_instances(num_instances=3, base_port=11434):
    """
    Run multiple Ollama instances in parallel

    Args:
        num_instances (int): Number of Ollama instances to run
        base_port (int): Starting port number
    """
    # List to store process objects
    processes = []

    try:
        # Create and start processes
        for i in range(num_instances):
            port = base_port + i
            # Optional: You could pass different models to different instances
            # models = ['llama2', 'mistral', 'gemma']
            # p = multiprocessing.Process(target=run_ollama_instance, args=(port, models[i]))
            p = multiprocessing.Process(target=run_ollama_instance, args=(port,))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating Ollama instances...")
        # Terminate all processes
        for p in processes:
            p.terminate()
            p.join()

    except Exception as e:
        print(f"Error running Ollama instances: {e}")
        # Ensure all processes are terminated
        for p in processes:
            p.terminate()
            p.join()


def main():
    # Number of Ollama instances to run
    num_instances = 3
    base_port = 11434

    print(f"Starting {num_instances} Ollama instances")
    run_multiple_ollama_instances(num_instances, base_port)


if __name__ == '__main__':
    main()
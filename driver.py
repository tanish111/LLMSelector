import os
import re
import subprocess
import time
# Read input.csv and populate query_to_category
query_to_category = {}
with open("input.csv", "r") as file:
    pattern = re.compile(r'"([^"]+)","([^"]+)"')
    for line in file:
        match = pattern.search(line)
        if match:
            query, category = match.groups()
            query_to_category[query] = category

# Create pipes
to_child_read, to_child_write = os.pipe()
from_child_read, from_child_write = os.pipe()

# Fork the process
pid = os.fork()

if pid == 0:
    # Child process
    os.dup2(to_child_read, 0)  # Redirect stdin
    os.dup2(from_child_write, 1)  # Redirect stdout

    os.close(to_child_write)
    os.close(from_child_read)

    try:
        os.execlp("python3", "python3", "main.py")
    except Exception as e:
        print(f"Error executing child process: {e}")
    os._exit(1)

else:
    # Parent process
    os.close(to_child_read)
    os.close(from_child_write)

    llm_maps = {
        "history": [1, 0, 2],
        "economics": [2, 1, 0],
        "creativity": [1, 0, 2],
        "coding": [0, 1, 2],
        "literature": [1, 0, 2],
        "physics": [0, 1, 2],
        "science": [0, 2, 1],
    }

    output = ""
    buffer_size = 1024
    while True:
        buffer = os.read(from_child_read, buffer_size).decode()
        if buffer:
            output += buffer
            break
        time.sleep(0.5)
    while True:
        query_regex = re.compile(r"Query:- \$ (.+?) \$")
        first_response_regex = re.compile(r"First-Response:- \$ (.+?) \$")
        second_response_regex = re.compile(r"Second-Response:- \$ (.+?) \$")
        query = ""
        first_response = ""
        second_response = ""
        cluster_regex = re.compile(r"\$-----Cluster-Started-----\$(.*?)\$-----Cluster-Ended-----\$", re.DOTALL)
        cluster_match = cluster_regex.search(output)
        if cluster_match:
            cluster_text = cluster_match.group(1)
            print(f"Found cluster text: {cluster_text}")
            # output = output[cluster_match.end():]
            # Dump output in cluster.log
            with open("cluster.log", "w") as log_file:
                log_file.write(cluster_text + "\n")
        match = query_regex.search(output)
        if match:
            query = match.group(1)
            print(f"Found query: {query}")
            output = output[match.end():]

        match = first_response_regex.search(output)
        if match:
            first_response = match.group(1)
            print(f"Found first response: {first_response}")
            output = output[match.end():]

        match = second_response_regex.search(output)
        if match:
            second_response = match.group(1)
            print(f"Found second response: {second_response}")
            output = output[match.end():]

        # Get the category for the current query
        category = query_to_category.get(query, "")

        # Get the desired LLM order for the category
        desired_llms = llm_maps.get(category, [0, 1, 2])

        # Compare responses to desired LLMs and decide input to child
        if first_response == str(desired_llms[0]):
            input_to_child = "A\n"
        elif second_response == str(desired_llms[0]):
            input_to_child = "B\n"
        elif first_response == str(desired_llms[1]):
            input_to_child = "A\n"
        elif second_response == str(desired_llms[1]):
            input_to_child = "B\n"
        elif first_response == str(desired_llms[2]):
            input_to_child = "A\n"
        elif second_response == str(desired_llms[2]):
            input_to_child = "B\n"
        else:
            # Default action if no match is found
            input_to_child = "A\n"

        # Write the decision to the child process
        os.write(to_child_write, input_to_child.encode())
        # Sleep until the process sends a response
        while True:
            buffer = os.read(from_child_read, buffer_size).decode()
            if buffer:
                output += buffer
                break
            time.sleep(1)
        time.sleep(0.5)
        # Check for the "Do you want to continue?" prompt
        continue_regex = re.compile(r"Do you want to continue\? \(Y/N\):")
        if continue_regex.search(output):
            user_input = input("Do you want to continue? (Y/N): ")
            os.write(to_child_write, (user_input + "\n").encode())
            if user_input.lower() == 'n':
                break

            query_input = input("Enter a query: ")
            os.write(to_child_write, (query_input + "\n").encode())

            output = ""
            while True:
                buffer = os.read(from_child_read, buffer_size).decode()
                if buffer:
                    output += buffer
                    break
                time.sleep(1)
            buffer = os.read(from_child_read, buffer_size).decode()
            if not buffer:
                break
            output += buffer

            selected_llm_regex = re.compile(r"Selected LLM: (.+?)")
            cluster_id_regex = re.compile(r"Cluster: (.+?)")

            match = selected_llm_regex.search(output)
            if match:
                selected_llm = match.group(0)
                print("Selected LLM: " , selected_llm)
                output = output[match.end():]

            match = cluster_id_regex.search(output)
            if match:
                cluster_id = match.group(0)
                print("Cluster: ",cluster_id)
                output = output[match.end():]
            response_regex = re.compile(r"Response from LLM: \$ (.+?) \$")
            match = response_regex.search(output)
            if match:
                response = match.group(1)
                print("Response from LLM: ", response)
                output = output[match.end():]
            rating = input("On a scale of 1 to 5, how would you rate the response? ")
            os.write(to_child_write, (rating + "\n").encode())
    os.close(to_child_write)
    os.close(from_child_read)

    # Wait for the child process to finish
    os.waitpid(pid, 0)

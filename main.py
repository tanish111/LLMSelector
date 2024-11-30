from LLMSelector import LLMSelectorEnv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
import random
import asyncio

async def update_clusters_async(env, queries):
    await asyncio.to_thread(env.update_clusters, queries)
    
# Function to call LLMs using LangChain
def call_llm_api(llm, query):
    return llm

# Collect user feedback
def collect_user_feedback(response_a, response_b):
    print("Response from LLM A:")
    print(response_a)
    print("Response from LLM B:")
    print(response_b)
    choice = input("Which response do you prefer? (A/B): ").strip().upper()
    return 0 if choice == "A" else 1

# Training loop
def train_llm_selector(env, queries, episodes=3):
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        for query in queries:
            print("Query:- $", query, "$")
            state = env.get_state(query)
            action = env.choose_action(state)

            # Get responses from selected LLMs
            llm_a = env.llms[env.llm_names[action[0]]]
            llm_b = env.llms[env.llm_names[action[1]]]
            response_a = "First-Response:- $ " + str(call_llm_api(llm_a, query)) + " $"
            response_b = "Second-Response:- $ " + str(call_llm_api(llm_b, query)) + " $"

            # Collect user feedback
            user_choice = collect_user_feedback(response_a, response_b)
            reward = 1 if user_choice == 0 else 0

            # Update Q-table
            env.update_q_table(state, action, reward)

        # Decay epsilon for exploration
        env.epsilon = max(env.epsilon * env.epsilon_decay, env.epsilon_min)
        env.print_q_table()

def ask_user_query(env,queries):
    query = input("Enter a query: ")
    print("Query:- $", query, "$")
    queries.append(query)
    state = env.get_state(query)
    action = env.get_best_model(state)
    print("Selected LLM:", env.llm_names[action])
    print("Cluster:", state+1)
    llm  = env.llms[env.llm_names[action]]
    response = call_llm_api(llm, query)
    print("Response from LLM: $ "+str(response) + " $")
    collect_user_feedback = input("On a scale of 1 to 5, how would you rate the response? ")
    env.update_q_table(state, [action], (int(collect_user_feedback)-2.5)/2.5)
    # env.update_q_table(state, action, int(collect_user_feedback)/5)
    
# Main script
if __name__ == "__main__":
    # Initialize LLMs using LangChain
    costs = [1,0.1,0.01]
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # if "GOOGLE_API_KEY" not in os.environ:
    #     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    # if "OPENAI_API_KEY" not in os.environ:
    #     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    # llms = {
    #     "OpenAI": ChatOpenAI(model="gpt-3.5-turbo"),
    #     "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
    #     "Ollama": ChatOllama(model="llama3:8b")
    # }
    llms = {
        "OpenAI": 0,
        "ChatGoogleGenerativeAI": 1,
        "Ollama": 2
    }
    queries = [
        "What are the main differences between a monarchy and a democracy?",
        "Explain the concept of Schrödinger's cat in quantum physics.",
        "Write a short story about a time traveler who visits ancient Egypt.",
        "What are the economic implications of AI on global job markets?",
        "Describe the lifecycle of a butterfly.",
        "Assume you are Sherlock Holmes and explain how you deduced a mystery.",
        "What is the capital city of Japan, and what is its cultural significance?",
        "Code in Python to find the maximum profit in the stock buy-and-sell problem.",
        "Who was Cleopatra, and why is she considered a significant historical figure?",
        "Compose a haiku about the beauty of autumn.",
        "What are the psychological effects of social media on teenagers?",
        "Explain the working principles of blockchain technology.",
        "Describe the process of nuclear fusion in stars.",
        "Write a dialogue between Romeo and Juliet set in the modern world.",
        "What are the primary causes and consequences of global warming?",
        "Who is Elon Musk, and what is his impact on the tech industry?",
        "Code in JavaScript to implement a basic to-do list application.",
        "What were the key events of World War II?",
        "Write a limerick about a curious cat.",
        "Explain the significance of the discovery of penicillin.",
        "What are the cultural differences between Eastern and Western societies?",
        "Describe the plot of 'Hamlet' in simple terms.",
        "What is the significance of photosynthesis in the Earth's ecosystem?",
        "Code in C++ to solve the Longest Common Subsequence problem.",
        "Who is Nelson Mandela, and what did he achieve during his lifetime?",
        "Write a persuasive essay on the importance of renewable energy.",
        "How do cryptocurrencies like Bitcoin function?",
        "Describe the history and evolution of space exploration.",
        "What is the Pythagorean theorem, and how is it applied in geometry?",
        "Write a motivational speech as though you are Martin Luther King Jr.",
        "Explain the principles of supply and demand in economics.",
        "What are the benefits of mindfulness meditation?",
        "Who was Mahatma Gandhi, and what role did he play in India's independence?",
        "Code in Python to calculate the factorial of a number using recursion.",
        "What is the purpose of the United Nations, and how does it function?",
        "Write a poem about the resilience of the human spirit.",
        "What are the ethical implications of cloning technology?",
        "Describe the structure and function of the human heart.",
        "Assume you are Hamlet and explain your thoughts on revenge.",
        "What is the theory of evolution, and who proposed it?",
        "Code in C to reverse a linked list.",
        "Who was Saddam Hussein, and what was his role in Iraq’s history?",
        "What are the cultural and historical highlights of the Renaissance?",
        "Write a song lyric about unrequited love.",
        "Explain the process of rain formation in the water cycle.",
        "What is the difference between machine learning and deep learning?",
        "Who is Kamala Harris, and what are her contributions to politics?",
        "Code in Java to implement a binary search algorithm.",
        "What are the effects of deforestation on biodiversity?",
        "Write a short story about a robot developing human emotions."
    ]
    env = LLMSelectorEnv(llms,costs=costs)
    env.fit_clusters(queries)
    env.print_clusters(queries)
    # env.visualize_embeddings()
    # train_llm_selector(env, queries, episodes=3)
    do_you_continue = True
    while(do_you_continue):
        do_you_continue = input("Do you want to continue? (Y/N): ").strip().upper() == "Y"
        if not do_you_continue:
            break
        ask_user_query(env,queries=queries)
        asyncio.run(update_clusters_async(env,queries=queries))
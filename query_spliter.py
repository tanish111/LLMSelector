import google.generativeai as genai

# Configure the API key
genai.configure(api_key="API_Key")

# Function for query decomposition using Gemini
def query_decomposition_gemini(question: str) -> str:
    """
    Sends the question to the Gemini model and gets a response.

    Args:
        question (str): The user's query.

    Returns:
        str: The decomposed query or an error message.
    """
    # Construct the prompt
    prompt = (
        "You are a helpful assistant that prepares queries that will be sent to a search component.",
        "Sometimes, these queries are very complex.",
"Your job is to simplify complex queries into multiple queries that can be answered in isolation to eachother.",
"If the query is simple, then keep it as it is.",
"Examples",
"1. Query: Did Microsoft or Google make more money last year?",
 "  Decomposed Questions: 'How much profit did Microsoft make last year? \n' 'How much profit did Google make last year?'\n",
"2. Query: What is the capital of France?",
   "Decomposed Questions: 'What is the capital of France?\n'",
"3. Query: {{question}}",
   "Decomposed Questions:",
        "You are an expert at converting user questions into database queries.\n",
        "Perform query decomposition. Given a user question, break it down into distinct sub-questions \n",
        "that you need to answer to address the original question. \n",
        "If there are acronyms or words you are not familiar with, do not try to rephrase them.\n\n"
        f"User: {question}\n\nAssistant:"
    )
    
    # Use the Gemini model to generate content
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")  # Specify the model version
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No response received."
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    # User question
    user_question = input("Enter a query: ")

    # Get the response from Gemini
    result = query_decomposition_gemini(user_question)

    print("Original Query:")
    print(user_question)
    print(result)

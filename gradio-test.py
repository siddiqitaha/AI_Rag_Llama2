import gradio as gr
import embedding_RetrievalQA_Wiki
import prompt

def get_answer_with_temp(question, temperature, history=''):
    embedding_RetrievalQA_Wiki.set_temperature(temperature)
    if history:
        full_prompt = f"{history}\nUser: {question}\nBot:"
    else:
        full_prompt = f"User: {question}\nBot:"
    answer = prompt.get_answer(full_prompt, temperature)  # Adjust based on your model's function signature
    new_history = f"{history}\nUser: {question}\nBot: {answer}"
    return answer, new_history  # Return the answer and the updated state

# Define the function to be used for the interface
def chat_interface(question, temperature=0.5, history=''):
    answer, new_history = get_answer_with_temp(question, temperature, history)
    return f"{new_history}", f"{new_history}"  # Return the new history as both the answer and the updated state

# Create the Gradio interface
frontend = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your question here...", label="Your Question"),
        gr.Slider(minimum=0, maximum=10, step=1, value=5, label="Creativity Level"),
        gr.State()  # State input to keep track of the conversation
    ],
    outputs=[
        gr.Textbox(label="Answer"),  # Main output for displaying the chatbot's response
        gr.State()  # State output to update and maintain the conversation history across submissions
    ],
    title="Locally Hosted LLM ChatBot",
    description="Enter a question to get an answer. Adjust the model's temperature to control randomness.",
    theme="default",
    css=".gradio-toolbar { display: none; }"  # Additional styling can be added as needed
)

# Launch the Gradio app
frontend.launch(
    inbrowser=True,
    share=False,
    auth=("username", "password"),
    auth_message="Please enter the Username and Password provided to you.",
    favicon_path="./images/favicon.jpg"
)
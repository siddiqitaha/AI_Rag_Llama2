import gradio as gr
import embedding_RetrievalQA_Wiki
import prompt
import css

def get_answer_with_temp(question, temperature):
    embedding_RetrievalQA_Wiki.set_temperature(temperature)
    answer = prompt.get_answer(question, temperature)  # Ensure this matches the expected function signature in prompt.py
    return answer

frontend = gr.Interface(
    fn=get_answer_with_temp,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your question here", label="Question"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Creativity Level (Temperature)")
    ],
    outputs=gr.Textbox(label="Answer", lines=5, placeholder="Due to CPU inference, please Allow time to process"),
    title="Locally Hosted LLM ChatBot",
    theme="Monochrome",
    description="Enter a question to get an answer. Adjust the model's temperature to control randomness.",
    css=css.custom_css
)

frontend.launch(
    inbrowser=True,
    share=True,
    auth=("username", "password"),
    auth_message="Please enter the Username and Password provided to you.",
    favicon_path="./images/favicon.jpg"
)
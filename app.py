import gradio as gr
from huggingface_hub import InferenceClient

# Custom background CSS with forced title styling
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP&family=Playfair+Display&display=swap');

body {
  background-image: url('https://cdn-uploads.huggingface.co/production/uploads/67351c643fe51cb1aa28f2e5/wuyd5UYTh9jPrMJGmV9yC.jpeg');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}
.gradio-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 100vh;
  padding-top: 2rem;
  padding-bottom: 2rem;
}  
#chat-panel {
  background-color: rgba(255, 255, 255, 0.85);
  padding: 2rem;
  border-radius: 12px;
  max-width: 700px;
  height: 70vh;
  margin: auto;
  box-shadow: 0 0 12px rgba(0, 0, 0, 0.3);
  overflow-y: auto;
}
.gradio-container .chatbot h1 {
  color: var(--custom-title-color) !important;
  font-family: 'Playfair Display', serif !important;
  font-size: 5rem !important;
  font-weight: bold !important;
  text-align: center !important;
  margin-bottom: 1.5rem !important;
  width: 100%;
}

/* Fallback font import */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');
"""

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

with gr.Blocks(css=css) as demo:
    # Title Markdown block
    gr.Markdown("French Tutor", elem_id="custom-title")
    
    with gr.Column(elem_id="chat-panel"):
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            system_message = gr.Textbox(
                value="You are a helpful French tutor.", 
                label="System Message"
            )
            max_tokens = gr.Slider(
                minimum=1, 
                maximum=2048, 
                value=512, 
                step=1, 
                label="Response Length"
            )
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=4.0, 
                value=0.7, 
                step=0.1, 
                label="Creativity"
            )
            top_p = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.95, 
                step=0.05, 
                label="Dynamic Text"
            )
        
        gr.ChatInterface(
            respond,
            additional_inputs=[
                system_message, 
                max_tokens, 
                temperature, 
                top_p
            ]
        )

if __name__ == "__main__":
    demo.launch()

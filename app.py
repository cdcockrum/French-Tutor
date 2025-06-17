import gradio as gr
from huggingface_hub import InferenceClient

# Background CSS
css = """
body {
  background-image: url('https://cdn-uploads.huggingface.co/production/uploads/67351c643fe51cb1aa28f2e5/YcsJnPk8HJvXiB5WkVmf1.jpeg');
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
#title-container {
  background-color: rgba(255, 255, 255, 0.85);  /* match chat panel */
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin: 2rem 0;
  width: fit-content;
  max-width: 500px;
  text-align: left;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  margin-left: 0rem;  /* Aligns left edge */
}
#title-container h1 {
  color: #222 !important;
  font-size: 4rem;
  font-family: 'Noto Sans JP', sans-serif;
  margin: 0;
}
#title-container .subtitle {
  font-size: 1.1rem;
  font-family: 'Noto Sans', sans-serif;
  color: #222 !important;
  margin-top: 0.5rem;
  margin-bottom: 0;
  width: 100%;
  display: block;
}
#chat-panel {
  background-color: rgba(255, 255, 255, 0.85);
  padding: 2rem;
  border-radius: 12px;
  justify-content: center;
  width: 100%;
  max-width: 700px;
  height: 70vh;
  box-shadow: 0 0 12px rgba(0, 0, 0, 0.3);
  overflow-y: auto;
}
.gradio-container .chatbot h1 {
   color: var(--custom-title-color) !important;
   font-family: 'Noto Sans', serif !important;
   font-size: 5rem !important;
   font-weight: bold !important;
   text-align: center !important;
   margin-bottom: 1.5rem !important;
   width: 100%;
}
"""

# Model client (consider switching to a public model like mistralai if 401 persists)
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Level prompt selector
def level_to_prompt(level):
    return {
        "A1": "You are a friendly French tutor. Focus on the user's specific question. Use simple French and explain in English. If helpful, you may include word origins or cultural notes, but avoid unrelated tangents and voice features.",
        "A2": "You are a patient French tutor. Respond to the user's question clearly. You may include brief relevant background such as word origin, common mistakes, or cultural usage — but only if directly related to the question. Do not mention or suggest voice interaction.",
        "B1": "You are a helpful French tutor. Use mostly French and minimal English. You can add short on-topic insights (like grammar tips or usage context) but avoid unrelated vocabulary or tools.",
        "B2": "You are a French tutor. Respond primarily in French and include only concise, relevant elaborations. Avoid suggesting voice interaction or unrelated content.",
        "C1": "You are a native French tutor. Use fluent French and address only what was asked, but you may include brief cultural or historical context if directly relevant.",
        "C2": "You are a French language professor. Use sophisticated French to answer only the question. You may include historical or linguistic nuance but avoid speculation or tool suggestions."
    }.get(level, "You are a helpful French tutor.")


# Chat handler
def respond(message, history, user_level, max_tokens, temperature, top_p):
    system_message = level_to_prompt(user_level)
    messages = [{"role": "system", "content": system_message}]
    
    # Handle history
    if history and isinstance(history[0], tuple):
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    else:
        messages.extend(history)

    messages.append({"role": "user", "content": message})
    
    response = ""
    try:
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = msg.choices[0].delta.content
            if token:
                response += token
            yield response
    except Exception as e:
        yield f"Désolé! There was an error: {str(e)}"

# Gradio interface
with gr.Blocks(css=css) as demo:
    gr.HTML("""
    <div id="title-container">
      <h1>LE PROFESSEUR</h1>
      <p class="subtitle">French Tutor</p>
    </div>
    """)
    
    with gr.Column(elem_id="chat-panel"):
        with gr.Accordion("Advanced Settings", open=False):
            user_level = gr.Dropdown(
                choices=["A1", "A2", "B1", "B2", "C1", "C2"],
                value="A1",
                label="Your French Level (CEFR)"
            )
            max_tokens = gr.Slider(1, 2048, value=400, step=1, label="Response Length")
            temperature = gr.Slider(0.1, 4.0, value=0.5, step=0.1, label="Creativity")
            top_p = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Dynamic Text Sampling")

        gr.ChatInterface(
            respond,
            additional_inputs=[user_level, max_tokens, temperature, top_p],
            type="messages"
        )

if __name__ == "__main__":
    demo.launch()

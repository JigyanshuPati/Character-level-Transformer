import gradio as gr
import torch
from v2 import BigramLanguageModel

# Load the saved model
checkpoint = torch.load("harry_model.pt", map_location="cpu")
config = checkpoint['config']
char_to_int = checkpoint['vocab']['char_to_int']
int_to_char = checkpoint['vocab']['int_to_char']
block_size = checkpoint['vocab']['block_size']
vocab_size = checkpoint['vocab']['vocab_size']
device = "cpu"  # For Gradio deployment, keep on CPU for compatibility

# Redefine encode/decode using loaded vocab
def encode(text):
    return [char_to_int[c] for c in text if c in char_to_int]

def decode(ints):
    return ''.join([int_to_char[i] for i in ints])

# Load model and weights
model = BigramLanguageModel(vocab_size, block_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generation function for Gradio
def generate_from_prompt(prompt, max_new_tokens=200):
    idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        idx = model.generate(idx, max_new_tokens)
    return decode(idx[0].tolist()) 


animated_title_css = """
@keyframes glow {
  0%, 100% {
    text-shadow: 0 0 5px #ffcc00, 0 0 10px #ffcc00, 0 0 20px #ffcc00, 0 0 40px #ffa500, 0 0 80px #ffa500;
    color: #fff3bf;
  }
  50% {
    text-shadow: 0 0 10px #ffd700, 0 0 20px #ffd700, 0 0 30px #ffd700, 0 0 50px #ff8c00, 0 0 100px #ff8c00;
    color: #fff8dc;
  }
}

.glow {
  animation: glow 2s ease-in-out infinite;
  font-family: 'Harry P', serif;
  text-align: center;
}
"""

# Gradio Interface with Blocks for more control
with gr.Blocks(css=animated_title_css) as iface:
    gr.HTML("<h1 class='glow'>PotterGPT ✨</h1>")
    gr.Markdown("Enter a prompt and generate text using your custom-trained character-level transformer!")
    
    with gr.Row():
        # Left-hand side column for input
        with gr.Column(scale=1):
            prompt = gr.Textbox(lines=5, label="Prompt", placeholder="Enter your prompt here...")
            max_tokens = gr.Slider(50, 500, value=200, step=10, label="Max Tokens")
            generate_btn = gr.Button("Generate", variant="primary")
        
        # Right-hand side column for output
        with gr.Column(scale=1):
            output = gr.Textbox(lines=10, label="Generated Text", placeholder="Generated text will appear here...")
    
    generate_btn.click(
        fn=generate_from_prompt,
        inputs=[prompt, max_tokens],
        outputs=output
    )

if __name__ == "__main__":
    iface.launch()

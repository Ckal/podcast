#Using codes from killerz3/PodGen & eswardivi/Podcastify
#For ZeroGPU limit, I roll back to inference API. You can use local or HF model also, remove the relative comment sign, it works;
import json
import httpx
import os
import re
import asyncio
import edge_tts
import tempfile
import gradio as gr
from huggingface_hub import AsyncInferenceClient
from pydub import AudioSegment
#from transformers import AutoModelForCausalLM, AutoTokenizer

from moviepy.editor import AudioFileClip, concatenate_audioclips

system_prompt = '''
    You are an talkshow podcast generator. You have to create short conversations between Alice and Bob that gives an overview of the News given by the user.
    Please provide the script and output strictly in the following JSON format:
    {
      "title": "[string]",
      "content": {
        "Alice_0": "[string]",
        "BOB_0": "[string]",
        ...
      }
    }
    #Please note that the [string] you generate now must be in based on the tone of people's daily life.
    #No more than five rounds of conversation, be concise.
'''

DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Link to Podcast</h1>
<p>A podcast talking about the link's content you provided.</p>
<p>🔎 Paste a website link with http/https.</p>
<p>🦕 Now using inference API. Modify codes to use transformer.</p>
</div>
'''

css = """
h1 {
    text-align: center;
    display: block;
}
p {
    text-align: center;
}
footer {
    display:none !important
}
"""

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
Client = AsyncInferenceClient(MODEL_ID)


"""
# USING LOCAL MODEL
model = AutoModelForCausalLM.from_pretrained(
     MODEL_ID,
     torch_dtype=torch.float16,
     device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""


def validate_url(url):
    try:
        response = httpx.get(url, timeout=60.0)
        response.raise_for_status()
        return response.text
    except httpx.RequestError as e:
        return f"An error occurred while requesting {url}: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"Error response {e.response.status_code} while requesting {url}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def fetch_text(url):
    print("Entered Webpage Extraction")
    prefix_url = "https://r.jina.ai/"
    full_url = prefix_url + url
    print(full_url)
    print("Exited Webpage Extraction")
    return validate_url(full_url)

async def text_to_speech(text, voice, filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)


async def gen_show(script):
    title = script['title']
    content = script['content']

    temp_files = []

    tasks = []
    for key, text in content.items():
        speaker = key.split('_')[0]  # Extract the speaker name
        index = key.split('_')[1]    # Extract the dialogue index
        voice = "en-US-JennyNeural" if speaker == "Alice" else "en-US-GuyNeural"

        # Create temporary file for each speaker's dialogue
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_files.append(temp_file.name)

        filename = temp_file.name
        tasks.append(text_to_speech(text, voice, filename))
        print(f"Generated audio for {speaker}_{index}: {filename}")

    await asyncio.gather(*tasks)

    # Combine the audio files using moviepy
    audio_clips = [AudioFileClip(temp_file) for temp_file in temp_files]
    combined = concatenate_audioclips(audio_clips)

    # Create temporary file for the combined output
    output_filename = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name

    # Save the combined file
    combined.write_audiofile(output_filename)
    print(f"Combined audio saved as: {output_filename}")

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
        print(f"Deleted temporary file: {temp_file}")

    return output_filename
    
"""
# USING LOCAL MODEL
def generator(messages):
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt'
    )
    
    output_ids = model.generate(
        input_ids.to('cuda'), 
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096,
        temperature=0.5,
        repetition_penalty=1.2,
        )
    
    results = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(results)
    return results
"""


def extract_content(text):
    """Extracts the JSON content from the given text."""
    match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

async def main(link):
    if not link.startswith("http://") and not link.startswith("https://"):
        return "URL must start with 'http://' or 'https://'",None

    text = fetch_text(link)

    if "Error" in text:
        return text, None

    prompt = f"News: {text} json:"
    formatted_prompt = system_prompt + "\n\n\n" + prompt
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": prompt},
    # ]

    answer = await Client.text_generation(
        prompt=formatted_prompt, 
        max_new_tokens=4096,
        temperature=0.7,
        return_full_text=False)
    print(answer)
    #generated_script = extract_content(generator(messages))
    generated_script = extract_content(answer)
    print("Generated Script:"+generated_script)

    # Check if the generated_script is empty or not valid JSON
    if not generated_script or not generated_script.strip().startswith('{'):
        raise ValueError("Failed to generate a valid script.")

    script_json = json.loads(generated_script)  # Use the generated script as input
    output_filename = await gen_show(script_json)
    print("Output File:"+output_filename)

    # Read the generated audio file
    return output_filename

with gr.Blocks(theme='soft', css=css, title="Musen") as iface:
    with gr.Accordion(""):
        gr.Markdown(DESCRIPTION)
    with gr.Row():
        output_box = gr.Audio(label="Podcast", type="filepath", interactive=False, autoplay=True, elem_classes="audio")  # Create an output textbox
    with gr.Row():
        input_box = gr.Textbox(label="Link", placeholder="Enter a http link")
    with gr.Row():
        submit_btn = gr.Button("🚀 Send")  # Create a submit button
        clear_btn = gr.ClearButton(output_box, value="🗑️ Clear") # Create a clear button

    # Set up the event listeners
    submit_btn.click(main, inputs=input_box, outputs=output_box)


#gr.close_all()

iface.queue().launch(show_api=False)  # Launch the Gradio interface
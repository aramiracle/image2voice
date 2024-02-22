import gradio as gr
import scipy
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Initialize variables for loaded models
processor_image = None
model_image = None
processor_text = None
model_text = None

# Define language code to name mapping with flag emojis
language_mapping = {
    "🇺🇸 English": "eng",
    "🇮🇷 Persian": "pes",
    "🇷🇺 Russian": "rus",
    "🇫🇷 French": "fra",
    "🇩🇪 German": "deu",
    "🇪🇸 Spanish": "spa",
    "🇹🇷 Turkish": "tur",
    "🇯🇵 Japanese": "jpn",
    "🇮🇹 Italian": "ita",
    "🇰🇷 Korean": "kor"
}

def initialize_models(image_model_path, text_model_path):
    global processor_image, model_image, processor_text, model_text
    # Load image captioning models
    processor_image = BlipProcessor.from_pretrained(image_model_path)
    model_image = BlipForConditionalGeneration.from_pretrained(image_model_path)

    # Load text translation models
    processor_text = AutoProcessor.from_pretrained(text_model_path)
    model_text = SeamlessM4Tv2Model.from_pretrained(text_model_path)

def generate_audio_from_image(image,
                              tgt_lang,
                              output_path='out_from_text.wav',
                              src_lang="eng"):
    global processor_image, model_image, processor_text, model_text

    # Generate image caption
    inputs = processor_image(image, return_tensors="pt")
    out = model_image.generate(**inputs)
    text = processor_image.decode(out[0], skip_special_tokens=True)

    # Translate generated text
    text_inputs = processor_text(text=text, src_lang=src_lang, return_tensors="pt")
    output_tokens = model_text.generate(**text_inputs, tgt_lang=language_mapping[tgt_lang], generate_speech=False)
    translated_text = processor_text.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)


    # Save audio file
    audio_array_from_text = model_text.generate(**text_inputs, tgt_lang=language_mapping[tgt_lang])[0].cpu().numpy().squeeze()
    sample_rate = model_text.config.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array_from_text)

    return output_path, translated_text

if __name__ == "__main__":
    image_model_path = "Salesforce/blip-image-captioning-large"
    text_model_path = "facebook/seamless-m4t-v2-large"
    initialize_models(image_model_path, text_model_path)

    iface = gr.Interface(
    fn=generate_audio_from_image,
    inputs=["image",
            gr.Dropdown(
            choices=list(language_mapping.keys()), label="Language"
    )],
    outputs=["audio", "text"],
    title="🌟 Generate Audio from Image 📸🔊",
    description="""✨ Welcome to the Audio Generation from Image Tool! ✨\n
    This powerful tool seamlessly combines cutting-edge image captioning and text-to-speech technologies, offering you an immersive audiovisual experience like never before. Simply upload an image and select the source language to generate audio from the image caption. The generated audio will be in Persian (🇮🇷), offering you a unique perspective on your visual content.\n
    🎨 Explore New Dimensions of Visual Content 🚀\n
    Whether you're a language enthusiast looking to enhance your skills, a content creator seeking innovative ways to engage your audience, or simply curious about the capabilities of AI-driven technologies, this tool opens up endless possibilities. Witness your images come to life as the AI-powered models analyze and describe them, transforming static visuals into dynamic auditory experiences.\n
    🤖 Powered by Advanced AI Models 🤖\n
    Behind the scenes, sophisticated machine learning models drive this tool forward. Leveraging state-of-the-art image captioning models like Salesforce's "blip-image-captioning-large" and text-to-speech models such as Facebook's "seamless-m4t-v2-large", this interface delivers high-quality audio outputs with remarkable accuracy and clarity.\n
    🔊 Immerse Yourself in the World of AI-Generated Audiovisual Experiences 🔊\n
    Whether you're exploring new languages, crafting captivating multimedia presentations, or simply enjoying the fusion of art and technology, this tool invites you to dive into the realm of AI-generated audiovisual experiences. Let your imagination soar as you witness the convergence of image and sound, powered by the latest advancements in artificial intelligence.\n
    Supported output languages: English (🇺🇸), Persian (🇮🇷), Russian (🇷🇺), French (🇫🇷), German (🇩🇪), Spanish (🇪🇸), Turkish (🇹🇷), Japanese (🇯🇵), Italian (🇮🇹), Korean (🇰🇷)""",
    )
  
iface.launch()

import scipy
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, SeamlessM4Tv2Model

def load_models(image_model_path, text_model_path):
    # Load image captioning models
    processor_image = BlipProcessor.from_pretrained(image_model_path)
    model_image = BlipForConditionalGeneration.from_pretrained(image_model_path)

    # Load text translation models
    processor_text = AutoProcessor.from_pretrained(text_model_path)
    model_text = SeamlessM4Tv2Model.from_pretrained(text_model_path)

    return processor_image, model_image, processor_text, model_text

def generate_audio_from_image(image_path,
                              output_path = 'out_from_text.wav',
                              image_model_path = "Salesforce/blip-image-captioning-large",
                              text_model_path = "facebook/seamless-m4t-v2-large",
                              src_lang="eng"):
    processor_image, model_image, processor_text, model_text = load_models(image_model_path, text_model_path)

    # Load image
    raw_image = Image.open(image_path).convert('RGB')

    # Generate image caption
    inputs = processor_image(raw_image, return_tensors="pt")
    out = model_image.generate(**inputs)
    text = processor_image.decode(out[0], skip_special_tokens=True)

    # Translate generated text
    text_inputs = processor_text(text=text, src_lang=src_lang, return_tensors="pt")
    audio_array_from_text = model_text.generate(**text_inputs, tgt_lang="pes")[0].cpu().numpy().squeeze()

    # Save audio file
    sample_rate = model_text.config.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array_from_text)

if __name__ == "__main__":
    output_path = 'out_from_text.wav'
    image_path = 'young_days.png'
    generate_audio_from_image(image_path, output_path)

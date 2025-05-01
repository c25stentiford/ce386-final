import sounddevice as sd
import numpy as np
import numpy.typing as npt
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from typing import Dict
from ollama import Client
from pydantic import BaseModel
import requests

LLM_MODEL: str = "gemma3:27b"  # Change this to be the model you want


def record_audio(duration_seconds: float = 0.75) -> npt.NDArray[np.float32]:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)


def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


client: Client = Client(
    host="http://ai.dfec.xyz:11434"  # Change this to be the URL of your LLM
)


class Place(BaseModel):
    place: str
    point: bool


def llm_parse_for_wttr(prompt: str):
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": """
                Your purpose is to extract the intended location from the user's weather request.
                The request is transcribed from speech by a different AI model and may contain errors.
                Take your time to ensure correct response. Place the extracted location into `place` as a string.
                
                The `point` field of the JSON output specifies if the location is a specific point, such as airports,
                train stations, landmarks, named spots, or anything else along those lines.
                In that case, set `point` to true.
                
                Places which are NOT points are things like cities, counties, states, countries, provinces, regions,
                military bases, departments, cantons, prefectures, arrondissements, boroughs, communes, districts, towns, 
                Census-Designated Places in the U.S., unincorporated communities, villages, tribal reservations, and neighborhoods.
                In any of these cases or similar, set `point` to false.
                
                Ensure the location is in a concise format, such as:
                "Paris", "Denver, Colorado", "Taoyuan, Taiwan", "New York City", "Osaka International Airport" or "Tokyo Skytree".
                
                If the given location appears to be a three-letter airport code, leave it intact.
                Examples of airport codes include: "FUK", "MUC", "BKK", "SFO", and "KHH".
                
                The transcription model may add extraneous punctuation into spoken airport codes. For instance, 
                "BKK" is sometimes transcribed as "BK.K." Look for occurences of similar errors and correct them.
                """,
            },
            {"role": "user", "content": prompt},
        ],
        model=LLM_MODEL,
        format=Place.model_json_schema(),
    )
    place = Place.model_validate_json(response.message.content)

    if place.point and not (
        place.place.upper() == place.place and len(place.place) == 3
    ):
        place.place = "~" + place.place

    return place.place.replace(" ", "+")


if __name__ == "__main__":
    model_id = "distil-whisper/distil-medium.en"
    print("Using model_id " + model_id)
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_pipeline(model_id, torch_dtype, device)
    print("Ready")

    while True:
        # print("Recording")
        audio = record_audio()
        # print("Transcribing")
        speech: Dict[str, str] = pipe(audio)
        # print(speech)
        if "computer" in speech["text"].lower():
            print("Recording prompt")
            speech = pipe(record_audio(5))
            location = llm_parse_for_wttr(speech["text"])
            try:
                wttr = requests.get("https://wttr.in/" + location)
                print(wttr.text)
            except:
                print("There was an error in retrieving weather data.")

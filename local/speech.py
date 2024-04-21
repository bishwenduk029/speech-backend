from melo.api import TTS
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel
import io

device = "cpu"
tts = TTS(language='EN', device=device)


class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = None
    voice: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[int] = None


app = FastAPI()


@app.post('/v1/audio/speech', response_class=StreamingResponse)
def speech(request_data: SpeechRequest):
    input = request_data.input
    speaker_ids = tts.hps.data.spk2id

    def audio_stream():
        bio = io.BytesIO()
        tts.tts_to_file(
            input, speaker_ids[request_data.voice], bio, speed=1, format='mp3')
        audio_data = bio.getvalue()
        yield audio_data

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

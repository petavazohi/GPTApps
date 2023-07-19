import os
import openai

key = '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'
openai.api_key = key
audio_file = open("audio.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)



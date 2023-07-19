import webrtcvad
import collections
import contextlib
import sys
import wave
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Generator, Tuple


def vad_audio(vad: webrtcvad.Vad, audio: np.ndarray, sr: int, frame_duration_ms: int = 30, padding_duration_ms: int = 300) -> Generator[bytes, None, None]:
    """
    Perform Voice Activity Detection (VAD) on the input audio.

    Parameters
    ----------
    vad : webrtcvad.Vad
        The VAD object initialized with a specific aggressiveness mode.
    audio : np.ndarray
        The input audio data as a numpy array.
    sr : int
        The sampling rate of the audio data.
    frame_duration_ms : int, optional
        The frame duration in milliseconds for the VAD. Defaults to 30.
    padding_duration_ms : int, optional
        The padding duration in milliseconds for the VAD. Defaults to 300.

    Yields
    ------
    bytes
        The voiced frames as bytes.

    Examples
    --------
    >>> import webrtcvad
    >>> import librosa
    >>> vad = webrtcvad.Vad(3)
    >>> audio, sr = librosa.load('recording.m4a')
    >>> voiced_frames = vad_audio(vad, audio, sr)
    >>> for frame in voiced_frames:
    >>>     print(frame)
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in audio:
        is_speech = vad.is_speech(frame.bytes, sr)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            if len(ring_buffer) == ring_buffer.maxlen:
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
        else:
            voiced_frames.append((frame, is_speech))
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f, speech in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f, speech in voiced_frames])


if __name__ == '__main__':
    # Load audio file
    audio, sr = librosa.load('recording.m4a')

    # Initialize VAD
    vad = webrtcvad.Vad(3)

    # Split the audio file
    chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=-16)

    # Save chunks to individual files
    for i, chunk in enumerate(chunks):
        chunk.export(f"chunk{i}.m4a", format="m4a")

import pyaudio

RESULTS_ROOT_PATH = "logs/"
CONF = "opensmile/config/emobase/emobase.conf"
# Настройки аудиозаписи
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

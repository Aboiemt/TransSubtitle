from __future__ import annotations
import argparse, json, sys, wave, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundcard as sc
from config import AppConfig
from audio_capture import SystemAudioCapture, list_loopback_devices
from asr import WhisperASR
from translator import DeepSeekTranslator
parser = argparse.ArgumentParser(description="Verify CPU ASR, WASAPI playback capture and mock streaming API integration. Plays the sample through the default speaker.")
parser.add_argument("--sample", type=Path, required=True, help="16 kHz mono signed 16-bit PCM Mandarin WAV containing 字幕")
parser.add_argument("--output", type=Path, default=Path("source-validation.json"))
args = parser.parse_args()
report={"python":sys.version.split()[0],"checks":{}}
with wave.open(str(args.sample),"rb") as wav:
    assert wav.getsampwidth()==2 and wav.getnchannels()==1 and wav.getframerate()==16000
    sample=np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2").astype(np.float32)/32768
cfg=AppConfig(whisper_model="base",whisper_device="cpu",whisper_language="zh")
asr=WhisperASR(cfg)
t=time.perf_counter(); asr.load(); report["checks"]["cpu_model_load"]={"passed":asr.ready,"seconds":round(time.perf_counter()-t,3),"model":asr.description}
t=time.perf_counter(); result=asr.transcribe(sample)
assert result.language=="zh" and "字幕" in result.text
report["checks"]["speech_recognition"]={"passed":True,"text":result.text,"audio_seconds":round(result.audio_seconds,3),"seconds":round(time.perf_counter()-t,3),"sample":"Windows SAPI synthetic Mandarin speech"}
chunks=[]
capture=SystemAudioCapture(cfg,on_audio=lambda x:chunks.append(x.copy()))
capture.start()
try:
    time.sleep(0.3)
    sc.default_speaker().play(sample,samplerate=16000)
    time.sleep(0.6)
finally: capture.stop()
assert chunks
recording=np.concatenate(chunks)
assert np.isfinite(recording).all() and float(np.max(np.abs(recording)))>0.001
result=asr.transcribe(recording)
assert "字幕" in result.text
report["checks"]["wasapi_loopback_to_whisper"]={"passed":True,"captured_seconds":round(len(recording)/16000,3),"text":result.text}
requests=[]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append({"path":self.path,"payload":payload})
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream; charset=utf-8")
        self.end_headers()
        for text in ("こんにちは。", "字幕翻訳へようこそ。"):
            chunk={"id":"verification","object":"chat.completion.chunk","created":0,"model":"verification","choices":[{"index":0,"delta":{"content":text},"finish_reason":None}]}
            self.wfile.write(("data: "+json.dumps(chunk,ensure_ascii=False)+"\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
try:
    config=AppConfig(deepseek_api_key="verification-only",deepseek_base_url=f"http://127.0.0.1:{server.server_port}/v1",deepseek_model="verification",translate_mode="zh2ja",translate_min_interval=0)
    translator=DeepSeekTranslator(config)
    deltas=[]
    translated=translator.translate(result.text,"zh",on_delta=deltas.append)
    assert translated.target_text=="こんにちは。字幕翻訳へようこそ。", translated
    assert translated.source_lang=="zh" and translated.target_lang=="ja"
    assert len(deltas)==2 and requests[-1]["path"]=="/v1/chat/completions"
    deltas.clear()
    simultaneous=translator.translate_simul("你好",[],on_delta=deltas.append)
    assert simultaneous.target_text=="こんにちは。字幕翻訳へようこそ。" and len(deltas)==2
    report["checks"]["streaming_api_contract"]={"passed":True,"requests":len(requests),"target":"ja","transport":"local mock OpenAI-compatible SSE server; not a translation-quality test","sentence_and_simultaneous":True}
finally: server.shutdown(); server.server_close()
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))

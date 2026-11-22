from __future__ import annotations
import asyncio, hmac, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Annotated
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

ROOT = Path(os.getenv("FACEFUSION_ROOT", "/facefusion")).resolve()
MAX_BYTES = max(1, int(os.getenv("MAX_UPLOAD_MB", "128"))) * 1024 * 1024
MAX_SECONDS = max(60, int(os.getenv("MAX_JOB_SECONDS", "1200")))
PROVIDER = os.getenv("FACEFUSION_EXECUTION_PROVIDER", "cuda").strip().lower()
MODELS = {"alphaface_256","blendswap_256","ghost_1_256","ghost_2_256","ghost_3_256","hififace_unofficial_256","hyperswap_1a_256","hyperswap_1b_256","hyperswap_1c_256","inswapper_128","inswapper_128_fp16","simswap_256","simswap_unofficial_512","uniface_256"}
PIXEL_BOOSTS = {"128x128","256x256","384x384","512x512","768x768","1024x1024"}
SELECTOR_MODES = {"reference","one","many"}
SELECTOR_ORDERS = {"left-right","right-left","top-bottom","bottom-top","small-large","large-small","best-worst","worst-best"}
GPU_LOCK = asyncio.Lock()
app = FastAPI(title="AI-CODE-CHAT FaceFusion Cloud Worker", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)

def require_key(x_api_key: Annotated[str | None, Header()] = None):
    expected = os.getenv("FACEFUSION_API_KEY", "").strip()
    if len(expected) < 32: raise HTTPException(503, "FACEFUSION_API_KEY must contain at least 32 characters")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected): raise HTTPException(401, "Invalid API key")

def sniff(data: bytes):
    if data.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
    if data.startswith(b"\xff\xd8\xff"): return "image/jpeg"
    if len(data)>=12 and data[:4]==b"RIFF" and data[8:12]==b"WEBP": return "image/webp"
    if len(data)>=12 and data[4:8]==b"ftyp": return "video/mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"): return "video/webm"
    return None

async def read_upload(upload: UploadFile, allowed: set[str]):
    data = await upload.read(MAX_BYTES + 1)
    if not data: raise HTTPException(400, "Empty upload")
    if len(data)>MAX_BYTES: raise HTTPException(413, f"Upload exceeds {MAX_BYTES // 1048576} MB")
    mime = sniff(data)
    if mime not in allowed: raise HTTPException(415, "Unsupported media type or invalid file content")
    return data, mime

@app.get("/health")
def health(): return {"ok": True, "service": "facefusion-cloud-worker"}

@app.get("/v1/info", dependencies=[Depends(require_key)])
def info():
    gpu, gpu_name = False, None
    try:
        result = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True,timeout=5)
        if result.returncode==0 and result.stdout.strip(): gpu, gpu_name = True, result.stdout.strip().splitlines()[0][:120]
    except (OSError, subprocess.TimeoutExpired): pass
    return {"ok":True,"service":"facefusion-cloud-worker","facefusion_version":os.getenv("FACEFUSION_VERSION","3.9.2"),"execution_provider":PROVIDER,"gpu_available":gpu,"gpu_name":gpu_name,"models":sorted(MODELS),"max_upload_mb":MAX_BYTES//1048576,"video_targets":["video/mp4","video/webm"]}

@app.post("/v1/swap", dependencies=[Depends(require_key)])
async def swap(source_face: Annotated[UploadFile,File()], target_media: Annotated[UploadFile,File()], face_swapper_model: Annotated[str,Form()]="hyperswap_1a_256", face_selector_mode: Annotated[str,Form()]="reference", face_selector_order: Annotated[str,Form()]="large-small", face_swapper_pixel_boost: Annotated[str,Form()]="256x256"):
    if not (ROOT/"facefusion.py").is_file(): raise HTTPException(503,"FaceFusion runtime is missing")
    if PROVIDER not in {"cpu","cuda","tensorrt","rocm","migraphx","directml","coreml"}: raise HTTPException(500,"Invalid execution provider")
    if face_swapper_model not in MODELS: raise HTTPException(400,"Unsupported FaceFusion model")
    if face_selector_mode not in SELECTOR_MODES or face_selector_order not in SELECTOR_ORDERS: raise HTTPException(400,"Invalid face selector setting")
    if face_swapper_pixel_boost not in PIXEL_BOOSTS: raise HTTPException(400,"Invalid pixel boost")
    face, fm = await read_upload(source_face,{"image/png","image/jpeg","image/webp"})
    target, tm = await read_upload(target_media,{"image/png","image/jpeg","image/webp","video/mp4","video/webm"})
    suffix={"image/png":".png","image/jpeg":".jpg","image/webp":".webp","video/mp4":".mp4","video/webm":".webm"}
    work=Path(tempfile.mkdtemp(prefix="aicc-facefusion-")); src=work/("source"+suffix[fm]); dst=work/("target"+suffix[tm]); out=work/("result"+suffix[tm])
    src.write_bytes(face); dst.write_bytes(target)
    cmd=[sys.executable,"facefusion.py","headless-run","--source-paths",str(src),"--target-path",str(dst),"--output-path",str(out),"--processors","face_swapper","--face-swapper-model",face_swapper_model,"--face-swapper-pixel-boost",face_swapper_pixel_boost,"--face-selector-mode",face_selector_mode,"--face-selector-order",face_selector_order,"--execution-providers",PROVIDER,"--log-level","info"]
    acquired=False
    try:
        await GPU_LOCK.acquire(); acquired=True
        proc=await asyncio.create_subprocess_exec(*cmd,cwd=str(ROOT),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        try: logs,_=await asyncio.wait_for(proc.communicate(),timeout=MAX_SECONDS)
        except asyncio.TimeoutError:
            proc.kill(); await proc.communicate(); raise HTTPException(504,f"FaceFusion exceeded {MAX_SECONDS} seconds")
        if proc.returncode!=0:
            detail=(logs or b"").decode("utf-8",errors="replace")[-1800:].strip()
            raise HTTPException(502,"FaceFusion failed"+(": "+detail if detail else ""))
        if not out.is_file() or out.stat().st_size==0: raise HTTPException(502,"FaceFusion produced no output")
        return FileResponse(str(out),media_type=tm,filename=out.name,background=BackgroundTask(shutil.rmtree,work,ignore_errors=True))
    except HTTPException:
        shutil.rmtree(work,ignore_errors=True); raise
    except Exception as exc:
        shutil.rmtree(work,ignore_errors=True); raise HTTPException(502,f"Worker error: {str(exc)[:300]}") from exc
    finally:
        if acquired: GPU_LOCK.release()

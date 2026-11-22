# FaceFusion Cloud Worker

Cloud GPU worker for AI-CODE-CHAT Studio. It wraps the official FaceFusion CLI with an authenticated FastAPI service.

## Endpoints
- GET /health: public health check.
- GET /v1/info: authenticated GPU and model information; requires x-api-key.
- POST /v1/swap: authenticated multipart upload with source_face and target_media; returns processed media.

## Build and publish
GitHub Actions builds and publishes ghcr.io/tanawutmalirum29-ux/facefusion-cloud-worker:latest and a commit-SHA tag when changes are pushed to main or the workflow is manually dispatched. After the first successful build, set the GHCR package visibility to Public if the GPU host needs anonymous image pulls.

## Deploy
Use an NVIDIA CUDA-capable GPU host. Set FACEFUSION_API_KEY to a random secret of at least 32 characters, FACEFUSION_EXECUTION_PROVIDER=cuda, MAX_UPLOAD_MB=128 and MAX_JOB_SECONDS=1200 in the host secret settings. Expose port 8080 over HTTPS and enter the same API key in AI-CODE-CHAT.

## Important limitation
The current endpoint is synchronous. Long-running videos can exceed an HTTP proxy timeout even when the worker timeout is longer. A persistent job queue plus object storage is required for robust long-video processing. Requests are serialized to reduce GPU out-of-memory risk.

Use only media and likenesses you have permission to process. Review FaceFusion and individual model licenses before commercial use.

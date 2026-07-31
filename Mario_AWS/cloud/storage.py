"""
cloud/storage.py -- S3 persistence for MarioOps.

Design rules:
  * If MARIOOPS_S3_BUCKET is unset (the local CPU test case), every
    method is a silent no-op and `enabled` is False. train.py / play.py
    never need to branch on "am I in the cloud?".
  * boto3 is imported lazily, so a local machine without boto3
    installed can still run training against local folders.
  * Gameplay videos get a shareable URL:
      - default: a presigned URL (works on private buckets; expires
        after MARIOOPS_S3_URL_EXPIRY seconds, max 7 days)
      - MARIOOPS_S3_PUBLIC_URLS=true: a permanent public object URL
        (requires a bucket policy that allows public GetObject on the
        runs/ prefix -- see README)
"""

from pathlib import Path

from config import (
    S3_BUCKET,
    S3_PREFIX,
    RUN_ID,
    AWS_REGION,
    S3_URL_EXPIRY_SECONDS,
    S3_PUBLIC_URLS,
    UPLOAD_FRAMES,
)


class S3Storage:
    def __init__(self):
        self.bucket = S3_BUCKET
        self.prefix = S3_PREFIX.strip("/")
        self.run_id = RUN_ID
        self._client = None

        if not self.bucket:
            self.enabled = False
            return

        try:
            import boto3  # imported lazily; not needed for local runs
            self._client = boto3.client("s3", region_name=AWS_REGION)
            self.enabled = True
        except Exception as exc:
            print(f"[s3] disabled (boto3 unavailable or misconfigured: {exc})")
            self.enabled = False

    # ------------------------------------------------------------ keys
    #
    # ALL keys are scoped under MARIOOPS_RUN_ID. Practical consequence:
    # a machine can only restore checkpoints uploaded under the SAME
    # run id. Training on AWS with MARIOOPS_RUN_ID=gpu-run-01 and then
    # running play.py locally with the default (local-dev) looks in
    # .../checkpoints/local-dev/... and silently finds nothing -- set
    # MARIOOPS_RUN_ID=gpu-run-01 locally to pull that run's artifacts.
    def checkpoint_key(self, filename: str) -> str:
        return f"{self.prefix}/checkpoints/{self.run_id}/{filename}"

    def log_key(self, filename: str) -> str:
        return f"{self.prefix}/logs/{self.run_id}/{filename}"

    def run_key(self, run_name: str, filename: str) -> str:
        return f"{self.prefix}/runs/{self.run_id}/{run_name}/{filename}"

    # ------------------------------------------------------- primitives
    def upload_file(self, local_path, key, content_type=None) -> bool:
        if not self.enabled:
            return False
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"[s3] skip upload; missing local file {local_path}")
            return False
        extra = {"ContentType": content_type} if content_type else None
        try:
            if extra:
                self._client.upload_file(
                    str(local_path), self.bucket, key, ExtraArgs=extra
                )
            else:
                self._client.upload_file(str(local_path), self.bucket, key)
            print(f"[s3] uploaded s3://{self.bucket}/{key}")
            return True
        except Exception as exc:
            print(f"[s3] upload failed for {key}: {exc}")
            return False

    def download_file(self, key, local_path) -> bool:
        if not self.enabled:
            return False
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, key, str(local_path))
            print(f"[s3] downloaded s3://{self.bucket}/{key} -> {local_path}")
            return True
        except Exception:
            # Missing key is normal on a first run -- not an error.
            return False

    # ------------------------------------------------ checkpoints / logs
    def upload_checkpoint(self, local_path) -> bool:
        return self.upload_file(
            local_path, self.checkpoint_key(Path(local_path).name)
        )

    def restore_checkpoint(self, local_path) -> bool:
        return self.download_file(
            self.checkpoint_key(Path(local_path).name), local_path
        )

    def restore_best_checkpoint(self, local_path) -> bool:
        """Same key derivation as restore_checkpoint -- the filename
        (mario_net_best.chkpt) maps to its own S3 object. Separate
        method purely for call-site readability."""
        return self.download_file(
            self.checkpoint_key(Path(local_path).name), local_path
        )

    def upload_training_log(self, local_path) -> bool:
        return self.upload_file(
            local_path, self.log_key(Path(local_path).name), content_type="text/csv"
        )

    # ------------------------------------------------------- public URLs
    def object_url(self, key: str) -> str:
        """
        Shareable URL for an object.

        Presigned by default (safe on private buckets, expires).
        Permanent public URL when MARIOOPS_S3_PUBLIC_URLS=true and the
        bucket policy allows anonymous GetObject on the runs/ prefix.
        """
        if not self.enabled:
            return ""
        if S3_PUBLIC_URLS:
            return f"https://{self.bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=S3_URL_EXPIRY_SECONDS,
            )
        except Exception as exc:
            print(f"[s3] could not presign {key}: {exc}")
            return ""

    # ------------------------------------------------------------- runs
    def upload_run(self, run_dir) -> str:
        """
        Upload a play.py run folder (clip + summary, frames optional)
        and return a shareable URL for the gameplay video ("" if the
        video is missing or S3 is disabled).
        """
        if not self.enabled:
            return ""
        run_dir = Path(run_dir)
        run_name = run_dir.name
        video_url = ""

        video = run_dir / "run.mp4"
        if video.exists():
            key = self.run_key(run_name, video.name)
            # video/mp4 makes browsers stream it instead of downloading.
            if self.upload_file(video, key, content_type="video/mp4"):
                video_url = self.object_url(key)

        summary = run_dir / "summary.json"
        if summary.exists():
            self.upload_file(
                summary, self.run_key(run_name, summary.name),
                content_type="application/json",
            )

        # Individual PNG frames are large and rarely needed in S3;
        # off by default to keep storage/transfer costs down.
        if UPLOAD_FRAMES:
            for frame in sorted((run_dir / "frames").glob("*.png")):
                self.upload_file(
                    frame, self.run_key(run_name, f"frames/{frame.name}"),
                    content_type="image/png",
                )

        return video_url

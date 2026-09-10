"""处理文件上传的路由。"""

import logging
import os
import stat

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_config
from deerflow.config.app_config import AppConfig
from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.sandbox.sandbox_provider import SandboxProvider, get_sandbox_provider
from deerflow.uploads.manager import (
    PathTraversalError,
    UnsafeUploadPathError,
    claim_unique_filename,
    delete_file_safe,
    enrich_file_listing,
    ensure_uploads_dir,
    get_uploads_dir,
    list_files_in_dir,
    normalize_filename,
    open_upload_file_no_symlink,
    upload_artifact_url,
    upload_virtual_path,
)
from deerflow.utils.file_conversion import CONVERTIBLE_EXTENSIONS, convert_file_to_markdown

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/threads/{thread_id}/uploads", tags=["uploads"])

UPLOAD_CHUNK_SIZE = 8192
DEFAULT_MAX_FILES = 10
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024
DEFAULT_MAX_TOTAL_SIZE = 100 * 1024 * 1024


class UploadedFileInfo(BaseModel):
    """上传和列表 API 暴露的上传文件元数据。"""

    filename: str
    size: int
    path: str
    virtual_path: str
    artifact_url: str
    extension: str | None = None
    modified: float | None = None
    original_filename: str | None = None
    markdown_file: str | None = None
    markdown_path: str | None = None
    markdown_virtual_path: str | None = None
    markdown_artifact_url: str | None = None


class UploadResponse(BaseModel):
    """文件上传的响应模型。"""

    success: bool
    files: list[UploadedFileInfo]
    message: str
    skipped_files: list[str] = Field(default_factory=list)


class UploadListResponse(BaseModel):
    """已上传文件列表的响应模型。"""

    files: list[UploadedFileInfo]
    count: int


class UploadLimits(BaseModel):
    """暴露给客户端的应用级上传限制。"""

    max_files: int
    max_file_size: int
    max_total_size: int


def _make_file_sandbox_writable(file_path: os.PathLike[str] | str) -> None:
    """确保上传的文件挂载到非本地沙箱后仍可写。

    在 AIO 沙箱模式下，gateway 先写入宿主侧的权威文件，
    之后沙箱运行时可能重写同一挂载路径。此处授予全局写权限，
    避免 gateway 用户与沙箱运行时用户之间的权限不匹配。
    """
    file_stat = os.lstat(file_path)
    if stat.S_ISLNK(file_stat.st_mode):
        logger.warning("Skipping sandbox chmod for symlinked upload path: %s", file_path)
        return

    writable_mode = stat.S_IMODE(file_stat.st_mode) | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH | stat.S_IRGRP | stat.S_IROTH
    chmod_kwargs = {"follow_symlinks": False} if os.chmod in os.supports_follow_symlinks else {}
    os.chmod(file_path, writable_mode, **chmod_kwargs)


def _make_file_sandbox_readable(file_path: os.PathLike[str] | str) -> None:
    """确保上传的文件对沙箱进程可读。

    对于 Docker 沙箱（AIO），gateway 以 root 身份写入文件（权限 0o600），
    然后将宿主目录 bind-mount 进容器。容器内的沙箱进程以非 root 用户
    运行，没有 group/other 读权限就无法读取这些文件。此函数添加
    ``S_IRGRP | S_IROTH``，使沙箱能读取上传内容。
    """
    file_stat = os.lstat(file_path)
    if stat.S_ISLNK(file_stat.st_mode):
        logger.warning("Skipping sandbox chmod for symlinked upload path: %s", file_path)
        return

    readable_mode = stat.S_IMODE(file_stat.st_mode) | stat.S_IRGRP | stat.S_IROTH
    chmod_kwargs = {"follow_symlinks": False} if os.chmod in os.supports_follow_symlinks else {}
    os.chmod(file_path, readable_mode, **chmod_kwargs)


def _uses_thread_data_mounts(sandbox_provider: SandboxProvider) -> bool:
    return bool(getattr(sandbox_provider, "uses_thread_data_mounts", False))


def _get_uploads_config_value(app_config: AppConfig, key: str, default: object) -> object:
    """读取 uploads 配置项，同时支持 dict 和属性两种访问方式。"""
    uploads_cfg = getattr(app_config, "uploads", None)
    if isinstance(uploads_cfg, dict):
        return uploads_cfg.get(key, default)
    return getattr(uploads_cfg, key, default)


def _get_upload_limit(app_config: AppConfig, key: str, default: int, *, legacy_key: str | None = None) -> int:
    try:
        value = _get_uploads_config_value(app_config, key, None)
        if value is None and legacy_key is not None:
            value = _get_uploads_config_value(app_config, legacy_key, None)
        if value is None:
            value = default
        limit = int(value)
        if limit <= 0:
            raise ValueError
        return limit
    except Exception:
        logger.warning("Invalid uploads.%s value; falling back to %d", key, default)
        return default


def _get_upload_limits(app_config: AppConfig) -> UploadLimits:
    return UploadLimits(
        max_files=_get_upload_limit(app_config, "max_files", DEFAULT_MAX_FILES, legacy_key="max_file_count"),
        max_file_size=_get_upload_limit(app_config, "max_file_size", DEFAULT_MAX_FILE_SIZE, legacy_key="max_single_file_size"),
        max_total_size=_get_upload_limit(app_config, "max_total_size", DEFAULT_MAX_TOTAL_SIZE),
    )


def _cleanup_uploaded_paths(paths: list[os.PathLike[str] | str]) -> None:
    for path in reversed(paths):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("Failed to clean up upload path after rejected request: %s", path, exc_info=True)


async def _write_upload_file_with_limits(
    file: UploadFile,
    *,
    uploads_dir: os.PathLike[str] | str,
    display_filename: str,
    max_single_file_size: int,
    max_total_size: int,
    total_size: int,
) -> tuple[os.PathLike[str] | str, int, int]:
    file_size = 0
    file_path, fh = open_upload_file_no_symlink(uploads_dir, display_filename)
    try:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            file_size += len(chunk)
            total_size += len(chunk)
            if file_size > max_single_file_size:
                raise HTTPException(status_code=413, detail=f"File too large: {display_filename}")
            if total_size > max_total_size:
                raise HTTPException(status_code=413, detail="Total upload size too large")
            fh.write(chunk)
    except Exception:
        fh.close()
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        raise
    else:
        fh.close()
    return file_path, file_size, total_size


def _auto_convert_documents_enabled(app_config: AppConfig) -> bool:
    """返回是否启用宿主侧自动文档转换。

    出于安全考虑默认关闭，除非运维在 config.yaml 中通过
    uploads.auto_convert_documents 显式开启。
    """
    try:
        raw = _get_uploads_config_value(app_config, "auto_convert_documents", False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    except Exception:
        return False


@router.post("", response_model=UploadResponse)
@require_permission("threads", "write", owner_check=True, require_existing=False)
async def upload_files(
    thread_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    config: AppConfig = Depends(get_config),
) -> UploadResponse:
    """向 thread 的 uploads 目录上传多个文件。"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    limits = _get_upload_limits(config)
    if len(files) > limits.max_files:
        raise HTTPException(status_code=413, detail=f"Too many files: maximum is {limits.max_files}")

    try:
        uploads_dir = ensure_uploads_dir(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    sandbox_uploads = get_paths().sandbox_uploads_dir(thread_id, user_id=get_effective_user_id())
    uploaded_files = []
    written_paths = []
    sandbox_sync_targets = []
    skipped_files = []
    total_size = 0
    # 跟踪本次请求内的文件名，避免重复的表单分片互相截断。
    # 对已有上传，单个替换上传仍保留历史上的覆盖行为。
    seen_filenames: set[str] = set()

    sandbox_provider = get_sandbox_provider()
    sync_to_sandbox = not _uses_thread_data_mounts(sandbox_provider)
    sandbox = None
    if sync_to_sandbox:
        sandbox_id = sandbox_provider.acquire(thread_id)
        sandbox = sandbox_provider.get(sandbox_id)
        if sandbox is None:
            raise HTTPException(status_code=500, detail="Failed to acquire sandbox")
    auto_convert_documents = _auto_convert_documents_enabled(config)

    for file in files:
        if not file.filename:
            continue

        try:
            original_filename = normalize_filename(file.filename)
            safe_filename = claim_unique_filename(original_filename, seen_filenames)
        except ValueError:
            logger.warning(f"Skipping file with unsafe filename: {file.filename!r}")
            continue

        try:
            file_path, file_size, total_size = await _write_upload_file_with_limits(
                file,
                uploads_dir=uploads_dir,
                display_filename=safe_filename,
                max_single_file_size=limits.max_file_size,
                max_total_size=limits.max_total_size,
                total_size=total_size,
            )
            written_paths.append(file_path)

            virtual_path = upload_virtual_path(safe_filename)

            if sync_to_sandbox:
                sandbox_sync_targets.append((file_path, virtual_path))

            file_info = {
                "filename": safe_filename,
                "size": file_size,
                "path": str(sandbox_uploads / safe_filename),
                "virtual_path": virtual_path,
                "artifact_url": upload_artifact_url(thread_id, safe_filename),
            }
            if safe_filename != original_filename:
                file_info["original_filename"] = original_filename

            logger.info(f"Saved file: {safe_filename} ({file_size} bytes) to {file_info['path']}")

            file_ext = file_path.suffix.lower()
            if auto_convert_documents and file_ext in CONVERTIBLE_EXTENSIONS:
                md_path = await convert_file_to_markdown(file_path)
                if md_path:
                    written_paths.append(md_path)
                    md_virtual_path = upload_virtual_path(md_path.name)

                    if sync_to_sandbox:
                        sandbox_sync_targets.append((md_path, md_virtual_path))

                    file_info["markdown_file"] = md_path.name
                    file_info["markdown_path"] = str(sandbox_uploads / md_path.name)
                    file_info["markdown_virtual_path"] = md_virtual_path
                    file_info["markdown_artifact_url"] = upload_artifact_url(thread_id, md_path.name)

            uploaded_files.append(file_info)

        except HTTPException as e:
            _cleanup_uploaded_paths(written_paths)
            raise e
        except UnsafeUploadPathError as e:
            logger.warning("Skipping upload with unsafe destination %s: %s", file.filename, e)
            skipped_files.append(safe_filename)
            continue
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            _cleanup_uploaded_paths(written_paths)
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")

    # 上传的文件以 0o600 权限创建（仅所有者可读写）。
    # 在 Docker 沙箱部署中，gateway 以 root 写入，但沙箱进程
    # 以非 root 用户（通常是 UID 1000）运行。没有 group/other
    # 读权限，沙箱就无法访问这些文件——无论 uploads 目录是
    # bind-mount 进容器还是通过 sandbox.update_file 同步。
    # 因此始终添加 group/other 读权限，使所有沙箱配置都能读取上传内容。
    for file_path in written_paths:
        _make_file_sandbox_readable(file_path)

    if sync_to_sandbox:
        for file_path, virtual_path in sandbox_sync_targets:
            _make_file_sandbox_writable(file_path)
            sandbox.update_file(virtual_path, file_path.read_bytes())

    message = f"Successfully uploaded {len(uploaded_files)} file(s)"
    if skipped_files:
        message += f"; skipped {len(skipped_files)} unsafe file(s)"

    return UploadResponse(
        success=not skipped_files,
        files=uploaded_files,
        message=message,
        skipped_files=skipped_files,
    )


@router.get("/limits", response_model=UploadLimits)
@require_permission("threads", "read", owner_check=True)
async def get_upload_limits(
    thread_id: str,
    request: Request,
    config: AppConfig = Depends(get_config),
) -> UploadLimits:
    """返回 gateway 对该 thread 使用的上传限制。"""
    return _get_upload_limits(config)


@router.get("/list", response_model=UploadListResponse)
@require_permission("threads", "read", owner_check=True)
async def list_uploaded_files(thread_id: str, request: Request) -> UploadListResponse:
    """列出 thread 的 uploads 目录中的所有文件。"""
    try:
        uploads_dir = get_uploads_dir(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = list_files_in_dir(uploads_dir)
    enrich_file_listing(result, thread_id)

    # gateway 额外附带沙箱相对路径。
    sandbox_uploads = get_paths().sandbox_uploads_dir(thread_id, user_id=get_effective_user_id())
    for f in result["files"]:
        f["path"] = str(sandbox_uploads / f["filename"])

    return UploadListResponse(**result)


@router.delete("/{filename}")
@require_permission("threads", "delete", owner_check=True, require_existing=True)
async def delete_uploaded_file(thread_id: str, filename: str, request: Request) -> dict:
    """从 thread 的 uploads 目录中删除文件。"""
    try:
        uploads_dir = get_uploads_dir(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return delete_file_safe(uploads_dir, filename, convertible_extensions=CONVERTIBLE_EXTENSIONS)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except PathTraversalError:
        raise HTTPException(status_code=400, detail="Invalid path")
    except Exception as e:
        logger.error(f"Failed to delete {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete {filename}: {str(e)}")

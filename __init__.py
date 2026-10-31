"""VNCCS - Visual Novel Character Creator Suite for ComfyUI."""

import importlib.util
import asyncio
import os, json
import sys
import traceback

def _runtime_module_available(name):
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

if (
    _runtime_module_available("torch")
    and _runtime_module_available("comfy")
    and _runtime_module_available("folder_paths")
):
    _nodes_module_name = f"{__name__}.nodes"
    _preloaded_nodes = sys.modules.get(_nodes_module_name)
    if _preloaded_nodes is not None and getattr(_preloaded_nodes, "__file__", None) is None:
        # Some custom-node loaders may pre-register this path as a namespace package.
        # Remove that placeholder so the regular relative import executes nodes/__init__.py.
        sys.modules.pop(_nodes_module_name, None)

    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    if not NODE_CLASS_MAPPINGS:
        raise RuntimeError("VNCCS node registration returned no nodes. Check the preceding registration traceback.")
else:
    # Package discovery and source-analysis tools do not provide a ComfyUI runtime.
    # Keep importing metadata possible without importing model dependencies such as torch.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "web"

def _vnccs_register_endpoint():  # lazy registration to avoid import errors in analysis tools
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return

    if getattr(PromptServer.instance, "app", None) is not None:
        from .nodes.http_state import install_cache_policy
        install_cache_policy(PromptServer.instance)

    @PromptServer.instance.routes.get("/vnccs/config")
    async def vnccs_get_config(request):
        name = request.rel_url.query.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)

        try:
            from .utils import config_path, ensure_safe_name
            name = ensure_safe_name(name, "character")
            cfg_path = config_path(name)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        if not os.path.exists(cfg_path):
            return web.json_response({"error": "not found"}, status=404)

        try:
            from .utils import character_storage_lock, character_dir
            def read_config():
                with character_storage_lock(character_dir(name)):
                    with open(cfg_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return web.json_response(data)
            return await asyncio.to_thread(read_config)
        except Exception as e:
            return web.json_response({"error": "read failed", "detail": str(e)}, status=500)

    @PromptServer.instance.routes.post("/vnccs/delete")
    async def vnccs_delete_character(request):
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=403)

        try:
            data = await request.json()
        except Exception:
            data = {}
        name = str(data.get("name") or request.rel_url.query.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        
        try:
            from .utils import character_dir, safe_join_under, base_output_dir, ensure_safe_name, character_storage_lock
            from .nodes.preview_runtime import run_preview_job
            from .nodes.generator_context import forget_character_contexts
            import shutil
            name = ensure_safe_name(name, "character")
            char_path = character_dir(name)
            safe_join_under(base_output_dir(), os.path.relpath(char_path, base_output_dir()))
            def remove_character():
                with character_storage_lock(char_path):
                    if not os.path.exists(char_path):
                        return False
                    shutil.rmtree(char_path)
                    forget_character_contexts(char_path)
                    return True

            if not await run_preview_job(remove_character):
                return web.json_response({"error": f"Character '{name}' not found"}, status=404)
            
            return web.json_response({"ok": True, "name": name, "deleted": True})
            
        except Exception as e:
            traceback.print_exc()
            return web.json_response({
                "error": "delete failed",
                "detail": str(e),
            }, status=500)

    @PromptServer.instance.routes.post("/vnccs/rename")
    async def vnccs_rename_character(request):
        """Rename a character: its folder, <name>_config.json and the name stored in its JSON files."""
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=403)
        try:
            data = await request.json()
        except Exception:
            data = {}
        try:
            from .utils import character_dir, ensure_safe_name, character_storage_lock
            from .nodes.preview_runtime import run_preview_job
            from .nodes.generator_context import forget_character_contexts
            old = ensure_safe_name(str(data.get("name") or "").strip(), "character")
            new = ensure_safe_name(str(data.get("new_name") or "").strip(), "character")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        if old == new:
            return web.json_response({"error": "new name is the same"}, status=400)

        def rename_character():
            src, dst = character_dir(old), character_dir(new)
            with character_storage_lock(src):
                if not os.path.isdir(src):
                    return f"Character '{old}' not found"
                if os.path.exists(dst):
                    return f"Character '{new}' already exists"
                os.rename(src, dst)
                forget_character_contexts(src)

            def swap(value):
                if isinstance(value, dict):
                    return {k: swap(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [swap(v) for v in value]
                return new if value == old else value

            config = os.path.join(dst, f"{old}_config.json")
            if os.path.exists(config):
                os.rename(config, os.path.join(dst, f"{new}_config.json"))
            for name in os.listdir(dst):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(dst, name)
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(swap(content), f, ensure_ascii=False, indent=2)
            return None

        try:
            error = await run_preview_job(rename_character)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": "rename failed", "detail": str(e)}, status=500)
        if error:
            return web.json_response({"error": error}, status=409 if "exists" in error else 404)
        return web.json_response({"ok": True, "name": new})

    @PromptServer.instance.routes.post("/vnccs/character_restore")
    async def vnccs_character_restore(request):
        """Write one file of a character (multipart: character, path, file) back to its folder.

        For clients that keep characters themselves (VNCCS Studio): before a run they put the
        config and the current images back, after it they remove the folder with /vnccs/delete.
        ``path`` is relative to the character folder, as /vnccs/character_files reports it.
        """
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=403)
        try:
            from .utils import (CHARACTER_FILE_EXTENSIONS, atomic_output_path, character_dir,
                                character_storage_lock, ensure_safe_name, safe_join_under)
            from .nodes.generator_context import forget_character_contexts
            post = await request.post()
            name = ensure_safe_name(str(post.get("character") or "").strip(), "character")
            rel = str(post.get("path") or "").strip().replace("\\", "/")
            upload = post.get("file")
            if not rel or rel.startswith("/") or any(part in ("", ".", "..") for part in rel.split("/")):
                raise ValueError("Invalid path")
            if os.path.splitext(rel)[1].lower() not in CHARACTER_FILE_EXTENSIONS:
                raise ValueError("Only character images and JSON files can be restored")
            if upload is None or not hasattr(upload, "file"):
                raise ValueError("file required")
            data = upload.file.read()
            root = character_dir(name)
            target = safe_join_under(root, *rel.split("/"))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        def write():
            with character_storage_lock(root):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with atomic_output_path(target) as temporary:
                    with open(temporary, "wb") as handle:
                        handle.write(data)
                forget_character_contexts(root)

        try:
            await asyncio.to_thread(write)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": "restore failed", "detail": str(e)}, status=500)
        return web.json_response({"ok": True, "path": rel, "size": len(data)})

    @PromptServer.instance.routes.post("/vnccs/studio_cleanup")
    async def vnccs_studio_cleanup(request):
        """Remove VNCCS Studio's temporary files: input/vnccs_studio/ and output/VNCCS_Studio/.

        Studio keeps the originals on the Mac and uploads them again when a run needs them.
        """
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=403)
        import shutil
        import folder_paths

        def clean():
            removed = 0
            for root in (os.path.join(folder_paths.get_input_directory(), "vnccs_studio"),
                         os.path.join(folder_paths.get_output_directory(), "VNCCS_Studio")):
                if not os.path.isdir(root):
                    continue
                for entry in os.listdir(root):
                    path = os.path.join(root, entry)
                    if os.path.isdir(path) and not os.path.islink(path):
                        removed += sum(len(files) for _, _, files in os.walk(path))
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                        removed += 1
            return removed

        try:
            removed = await asyncio.to_thread(clean)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": "cleanup failed", "detail": str(e)}, status=500)
        return web.json_response({"ok": True, "removed": removed})

    @PromptServer.instance.routes.post("/vnccs/create")
    @PromptServer.instance.routes.get("/vnccs/create")
    async def vnccs_create_character(request):
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=403)
        if getattr(request, "method", "GET") == "POST":
            try:
                data = await request.json()
                name = str(data.get("name", "")).strip()
            except (ValueError, TypeError, AttributeError):
                return web.json_response({"error": "Invalid create request"}, status=400)
        else:
            # Compatibility for older extensions; responses are explicitly no-store.
            name = request.rel_url.query.get("name", "").strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        try:
            from .utils import ensure_safe_name
            name = ensure_safe_name(name, "character")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        try:
            from .nodes.character_creator_v2 import CharacterCreatorV2
            catalog = data.get("catalog", "") if getattr(request, "method", "GET") == "POST" else ""
            return web.json_response(await asyncio.to_thread(CharacterCreatorV2.create_character, name, catalog))
        except Exception as e:
            traceback.print_exc()
            return web.json_response({
                "error": "create failed",
                "detail": str(e),
                "type": type(e).__name__,
            }, status=500)

    @PromptServer.instance.routes.get("/vnccs/create_costume")
    async def vnccs_create_costume(request):
        try:
            from .utils import validate_privileged_request
            validate_privileged_request(request)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=403)
        character_name = request.rel_url.query.get("character", "").strip()
        costume_name = request.rel_url.query.get("costume", "").strip()
        if not character_name or not costume_name:
            return web.json_response({"error": "character and costume required"}, status=400)
        try:
            from .utils import ensure_safe_name
            character_name = ensure_safe_name(character_name, "character")
            costume_name = ensure_safe_name(costume_name, "costume")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        try:
            from .utils import load_config, save_costume_info, ensure_costume_structure, character_dir, character_storage_lock
            def create_costume():
                with character_storage_lock(character_dir(character_name)):
                    config = load_config(character_name, strict=True)
                    if config is None:
                        config = {"character_info": {}, "costumes": {}}
                    costumes = config.setdefault("costumes", {})
                    if costume_name.casefold() in {"naked", "original"} or any(c.casefold() == costume_name.casefold() for c in costumes):
                        return web.json_response({"error": "Costume already exists"})
                    info = dict.fromkeys(("face", "head", "top", "bottom", "shoes", "negative_prompt"), "")
                    ensure_costume_structure(character_name, costume_name)
                    if not save_costume_info(character_name, costume_name, info):
                        return web.json_response({"error": "Failed to save"}, status=500)
                    return web.json_response({"ok": True, "costume": costume_name})
            return await asyncio.to_thread(create_costume)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

_vnccs_register_endpoint()

"""
ComfyUI API endpoint for SDScriptsTrainer sample presets.
"""

from __future__ import annotations

from .presets import get_preset_payload


def register_preset_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ModuleNotFoundError:
        return

    routes = PromptServer.instance.routes

    @routes.get("/sdscripts_trainer/presets")
    async def sdscripts_trainer_presets(_request):
        return web.json_response(get_preset_payload())

from __future__ import annotations

from ai_dev.command_groups import CommandHandlers


def build_command_handlers(**handlers) -> CommandHandlers:
    return CommandHandlers(
        command_init=handlers["command_init"],
        command_up=handlers["command_up"],
        command_down=handlers["command_down"],
        command_status=handlers["command_status"],
        command_pull_models=handlers["command_pull_models"],
        command_index=handlers["command_index"],
        command_retrieve=handlers["command_retrieve"],
        command_configure_cursor=handlers["command_configure_cursor"],
        command_models=handlers["command_models"],
        command_route_model=handlers["command_route_model"],
        command_spec_decode=handlers["command_spec_decode"],
        command_embed_enqueue=handlers["command_embed_enqueue"],
        command_embed_stats=handlers["command_embed_stats"],
        command_memory_explain=handlers["command_memory_explain"],
    )
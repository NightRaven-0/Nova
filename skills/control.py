# skills/control.py — Windows media & volume keys (no dependencies, via ctypes).

import sys

from skills.registry import skill

# Virtual-key codes for the keyboard's media/volume keys.
_MEDIA_VK = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
}


@skill(
    name="media_control",
    description="Control media playback and system volume using the keyboard media keys: play/pause, next, previous, stop, volume up/down, mute.",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play_pause", "next", "previous", "stop", "volume_up", "volume_down", "mute"],
            }
        },
        "required": ["action"],
    },
    examples=["pause the music", "next track", "turn it up", "mute"],
)
def media_control(args: dict) -> str:
    if not sys.platform.startswith("win"):
        return "Media control is only set up for Windows right now."
    action = (args.get("action") or "").strip().lower()
    vk = _MEDIA_VK.get(action)
    if vk is None:
        return f"Unknown media action: {action}"
    import ctypes

    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    return f"Done: {action.replace('_', ' ')}."

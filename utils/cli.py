# utils/cli.py

def print_banner() -> None:
    """Print Nova startup banner."""
    print(r"""
 _   _   ____   __      __    _    
| \ | | / __ \  \ \    / /   / \   
|  \| || |  | |  \ \  / /   / _ \  
| |\  || |  | |   \ \/ /   / ___ \ 
|_| \_| \____/     \__/   /_/   \_\
         N O V A   A S S I S T A N T
────────────────────────────────────────────
  • Say something and I'll listen.
  • Say "exit", "quit" or "stop" to leave.
────────────────────────────────────────────
""")

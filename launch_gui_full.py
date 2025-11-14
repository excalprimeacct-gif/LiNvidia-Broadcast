#!/usr/bin/env python3
"""
Launcher script for LiNvidia Broadcast full-featured GUI (audio + video)

This script can be run directly from anywhere.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and launch full GUI
from linvidia.ui.gui_full import launch_full_gui

if __name__ == '__main__':
    sys.exit(launch_full_gui())

#!/usr/bin/env python3
"""
Launcher script for LiNvidia Broadcast simple GUI (audio only)

This script can be run directly from anywhere.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and launch GUI
from linvidia.ui.gui import launch_gui

if __name__ == '__main__':
    sys.exit(launch_gui())

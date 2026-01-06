# coding:utf-8
"""
    :module: install_desktop.py
    :description: Create desktop file and copy supplied icon or install an app to Linux
    :author: Michel 'Mitch' Pecqueur
    :date: 2026.01
"""

import shutil
from pathlib import Path


def install_desktop(exe_path: Path, icon_file: Path | None = None, app_name: str | None = None) -> Path:
    """
    Create desktop file and copy supplied icon

    :param exe_path: Full path to the executable
    :param app_name: Application Name
    :param icon_file: .png icon file

    :return: Path to created desktop file
    """
    home = Path.home()
    desktop_dir = home / '.local/share/applications'
    icons_dir = home / '.local/share/icons/hicolor'

    desktop_dir.mkdir(parents=True, exist_ok=True)

    # Copy icon
    icon_path = None
    if icon_file.is_file():
        icon_path = icons_dir / f'apps/{exe_path.stem}.png'
        shutil.copy(icon_file, icon_path)

    # Write desktop file
    app_name = app_name or exe_path.stem

    desktop_file = desktop_dir / f'{app_name}.desktop'

    desktop_str = (f'[Desktop Entry]\n'
                   f'Type=Application\n'
                   f'Name={app_name}\n'
                   f'Exec={exe_path}\n')
    if icon_path:
        desktop_str += f'Icon={icon_path.stem}\n'
    desktop_str += f'Terminal=false\nCategories=Utility;'

    with open(desktop_file, 'w') as f:
        f.write(desktop_str)

    desktop_file.chmod(0o755)

    print(f"Installed {desktop_file}")

    return desktop_file

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from prompt_toolkit import Application, prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel

from data_designer.config.utils.constants import RICH_CONSOLE_THEME, NordColor

# Global padding configuration
LEFT_PADDING = 2
RIGHT_PADDING = 2

# Private console instance - all output goes through this
_console = Console(theme=RICH_CONSOLE_THEME)

# Public console alias for external imports
console = _console

# Command/input history for prompt_toolkit
_input_history = InMemoryHistory()

# Custom style for prompt_toolkit
_PROMPT_STYLE = Style.from_dict(
    {
        # Default text style
        "": "#ffffff",
        # Auto-suggestions in gray (from history)
        "auto-suggest": "fg:#666666",
        # Dim text for defaults and hints (Nord3 - lighter dark gray)
        "dim": "fg:#4c566a",
        # Selected completion in menu
        "completion-menu.completion.current": "bg:#88c0d0 #2e3440",
        "completion-menu.completion": "bg:#4c566a #d8dee9",
    }
)


# Sentinel value to indicate user wants to go back
class _BackSentinel:
    """Sentinel class to indicate user wants to go back to previous prompt."""

    def __repr__(self) -> str:
        return "BACK"


BACK = _BackSentinel()


def select_with_arrows(
    options: dict[str, str],
    prompt_text: str,
    default_key: str | None = None,
    allow_back: bool = False,
) -> str | None | _BackSentinel:
    """Interactive selection with arrow key navigation using inline menu.

    Uses prompt_toolkit's Application for an inline dropdown-style menu experience.

    Args:
        options: Dictionary of {key: display_text} options
        prompt_text: Prompt to display above options
        default_key: Default selected key (if None, first option is selected)
        allow_back: If True, adds a "Go back" option to return to previous prompt

    Returns:
        Selected key, None if cancelled, or BACK sentinel if user wants to go back
    """
    if not options:
        return None

    # Build list of keys and values
    keys = list(options.keys())
    back_key = "__back__"

    if allow_back:
        keys.append(back_key)
        options = {**options, back_key: "← Go back"}

    # Find default index
    if default_key and default_key in keys:
        selected_index = keys.index(default_key)
    else:
        selected_index = 0

    # Store result
    result = {"value": None, "cancelled": False}

    def get_formatted_text() -> list[tuple[str, str]]:
        """Generate the formatted text for the menu."""
        text = []
        # Add prompt with padding
        padding = " " * LEFT_PADDING
        text.append(("", f"{padding}{prompt_text}\n"))

        # Add options
        for i, key in enumerate(keys):
            display = options[key]
            if i == selected_index:
                # Selected item with Nord8 color
                text.append((f"fg:{NordColor.NORD8.value} bold", f"{padding}  → {display}\n"))
            else:
                # Unselected item
                text.append(("", f"{padding}    {display}\n"))

        # Add hint
        text.append(("fg:#666666", f"{padding}  (↑/↓: navigate, Enter: select, Esc: cancel)\n"))
        return text

    # Create key bindings
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")  # Ctrl+P
    def _move_up(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(keys)

    @kb.add("down")
    @kb.add("c-n")  # Ctrl+N
    def _move_down(event) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(keys)

    @kb.add("enter")
    def _select(event) -> None:
        result["value"] = keys[selected_index]
        event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")  # Ctrl+C
    def _cancel(event) -> None:
        result["cancelled"] = True
        event.app.exit()

    # Create the application
    app = Application(
        layout=Layout(
            HSplit(
                [
                    Window(
                        content=FormattedTextControl(get_formatted_text),
                        dont_extend_height=True,
                        always_hide_cursor=True,  # Hide cursor during menu selection
                    )
                ]
            )
        ),
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
    )

    try:
        # Run the application
        app.run()

        # Handle the result
        if result["cancelled"]:
            print_warning("Cancelled")
            return None
        elif result["value"] == back_key:
            print_info("Going back...")
            return BACK
        else:
            return result["value"]

    except (KeyboardInterrupt, EOFError):
        print_warning("Cancelled")
        return None


def prompt_text_input(
    prompt_msg: str,
    default: str | None = None,
    validator: Callable[[str], tuple[bool, str | None]] | None = None,
    mask: bool = False,
    completions: list[str] | None = None,
    allow_back: bool = False,
) -> str | None | _BackSentinel:
    """Prompt for text input with full line editing support.

    Supports standard keyboard shortcuts:
    - Ctrl+A: Move to beginning of line
    - Ctrl+E: Move to end of line
    - Ctrl+K: Delete to end of line
    - Ctrl+U: Delete entire line
    - Ctrl+W: Delete previous word
    - Arrow keys: Navigate character by character
    - Up/Down: Navigate through input history
    - Tab: Trigger completions (if provided)
    - Ctrl+C / Ctrl+D: Cancel

    Special inputs:
    - Type 'back' or 'b' to go back to previous prompt (if allow_back=True)

    Features:
    - Auto-suggestions from history (shown in gray)
    - Fuzzy completion from provided options
    - Persistent history across prompts

    Args:
        prompt_msg: Prompt message to display
        default: Default value if user presses Enter without input
        validator: Optional function that returns (is_valid, error_message)
        mask: If True, input is masked (for passwords/secrets)
        completions: Optional list of completion suggestions
        allow_back: If True, user can type 'back' to return to previous prompt

    Returns:
        User input string, None if cancelled, or BACK sentinel if user wants to go back
    """
    # Build the prompt text with padding
    # Use prompt-toolkit's HTML-like markup for colors
    padded_prompt = " " * LEFT_PADDING + prompt_msg
    if default:
        padded_prompt += f" <dim>(default: {default})</dim>"
    padded_prompt += ": "

    # Create completer if completions provided or if back is enabled
    completer = None
    if completions or allow_back:
        # Start with provided completions or empty list
        completion_list = list(completions) if completions else []
        # Add 'back' to completions if allowed
        if allow_back and "back" not in completion_list:
            completion_list.append("back")
        if completion_list:  # Only create completer if we have items
            word_completer = WordCompleter(completion_list, ignore_case=True)
            completer = FuzzyCompleter(word_completer)

    try:
        # Use prompt-toolkit for line editing with enhanced features
        # Wrap in HTML() to enable styled markup
        value = prompt(
            HTML(padded_prompt),
            default="",  # Empty input field
            is_password=mask,
            completer=completer,
            style=_PROMPT_STYLE,
            complete_while_typing=True,  # Show completions as you type
            auto_suggest=AutoSuggestFromHistory(),  # Show gray suggestions from history
            history=_input_history,  # Enable history with up/down arrows
            enable_history_search=False,  # Disable Ctrl+R search (optional)
        ).strip()

    except (KeyboardInterrupt, EOFError):
        print_warning("Cancelled")
        return None

    # Check if user wants to go back
    if allow_back and value.lower() in ("back", "b"):
        print_info("Going back...")
        return BACK

    # Use default if no input provided (user just pressed Enter)
    if not value and default:
        value = default

    # Skip validation if no input and no default
    if not value:
        return value

    # Validate if validator provided
    if validator:
        is_valid, error_msg = validator(value)
        if not is_valid:
            print_error(f"Error: {error_msg}")
            return prompt_text_input(prompt_msg, default, validator, mask, completions, allow_back)

    return value


def confirm_action(prompt_msg: str, default: bool = False) -> bool:
    """Prompt for yes/no confirmation with line editing support.

    Args:
        prompt_msg: Question to ask
        default: Default choice if user just presses Enter

    Returns:
        True for yes, False for no
    """
    default_text = "Y/n" if default else "y/N"
    padded_prompt = " " * LEFT_PADDING + f"{prompt_msg} <dim>[{default_text}]</dim>: "

    try:
        response = (
            prompt(
                HTML(padded_prompt),
                style=_PROMPT_STYLE,
                history=_input_history,
            )
            .strip()
            .lower()
        )
    except (KeyboardInterrupt, EOFError):
        print_warning("Cancelled")
        return False

    if not response:
        return default

    return response in ("y", "yes")


def display_config_preview(config: dict, title: str = "Configuration Preview") -> None:
    """Display a configuration dictionary as formatted YAML.

    Args:
        config: Configuration dictionary to display
        title: Title for the preview panel
    """
    import yaml

    yaml_str = yaml.safe_dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        allow_unicode=True,
    )

    # Calculate panel width to account for padding
    panel_width = _console.width - LEFT_PADDING - RIGHT_PADDING

    panel = Panel(
        yaml_str,
        title=title,
        title_align="left",
        border_style=NordColor.NORD14.value,
        width=panel_width,
    )
    _print_with_padding(panel)


def print_success(message: str) -> None:
    """Print a success message with green styling.

    Args:
        message: Success message to display
    """
    _print_with_padding(f"✅  {message}")


def print_error(message: str) -> None:
    """Print an error message with red styling.

    Args:
        message: Error message to display
    """
    _print_with_padding(f"❌  {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow styling.

    Args:
        message: Warning message to display
    """
    _print_with_padding(f"⚠️   {message}")


def print_info(message: str) -> None:
    """Print an info message with blue styling.

    Args:
        message: Info message to display
    """
    _print_with_padding(f"💡  {message}")


def print_text(message: str) -> None:
    """Print a text message.

    Args:
        message: Text message to display
    """
    _print_with_padding(message)


def print_header(text: str) -> None:
    """Print a styled header.

    Args:
        text: Header text
    """
    _console.print()
    # Create a manual rule that respects padding
    padding_str = " " * LEFT_PADDING
    available_width = _console.width - LEFT_PADDING - RIGHT_PADDING

    # Format the title text
    title_text = f" {text} "
    title_length = len(title_text)

    # Calculate how many rule characters on each side
    # (available_width - title_length) / 2
    rule_chars = max(0, (available_width - title_length) // 2)
    remaining = max(0, available_width - title_length - (rule_chars * 2))

    # Build the rule string
    rule_line = "─" * rule_chars + title_text + "─" * (rule_chars + remaining)

    # Print with padding and styling
    _console.print(f"{padding_str}[bold {NordColor.NORD8.value}]{rule_line}[/bold {NordColor.NORD8.value}]")
    _console.print()


def print_navigation_tip() -> None:
    """Display a concise navigation tip for interactive prompts."""
    tip = "[dim]Tip: Use arrow keys to navigate menus, type [bold]'back'[/bold] to edit previous entries, press [bold]Tab[/bold] for completions[/dim]"
    _print_with_padding(tip)
    _console.print()


def _print_with_padding(content: str | Panel) -> None:
    """Internal helper to print with left padding.

    Args:
        content: Content to print (string or Panel)
    """

    padding = " " * LEFT_PADDING
    if isinstance(content, Panel):
        # For panels, wrap in Rich's Padding to properly handle alignment
        padded_content = Padding(content, (0, 0, 0, LEFT_PADDING))
        _console.print(padded_content)
    else:
        # For strings, handle multi-line text
        if "\n" in str(content):
            lines = str(content).split("\n")
            for line in lines:
                _console.print(padding + line)
        else:
            _console.print(padding + str(content))

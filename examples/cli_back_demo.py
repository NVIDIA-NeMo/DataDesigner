#!/usr/bin/env python
"""
Demonstration of the 'back' functionality in Data Designer CLI.

This example shows how to use the BACK sentinel value to implement
navigation between prompts in a wizard-style interface.
"""

from data_designer.cli.interactive import BACK, prompt_text_input, select_with_arrows


def simple_wizard_with_back() -> None:
    """Demonstrate a simple wizard that supports going back."""
    print("=== Simple Wizard with Back Support ===\n")

    # Step 1: Get name
    step = 1
    name = None
    age = None
    color = None

    while True:
        if step == 1:
            # First prompt - no back option
            name = prompt_text_input("What is your name?", default="Alice", allow_back=False)
            if name is None:  # Cancelled
                print("Wizard cancelled!")
                return
            step = 2

        elif step == 2:
            # Second prompt - can go back to step 1
            age = prompt_text_input("What is your age?", default="25", allow_back=True)
            if age is None:  # Cancelled
                print("Wizard cancelled!")
                return
            elif age is BACK:  # Go back
                step = 1
                continue
            step = 3

        elif step == 3:
            # Third prompt - can go back to step 2
            color_options = {
                "red": "Red - Bold and vibrant",
                "blue": "Blue - Calm and serene",
                "green": "Green - Fresh and natural",
            }
            color = select_with_arrows(color_options, "Select your favorite color", allow_back=True)
            if color is None:  # Cancelled
                print("Wizard cancelled!")
                return
            elif color is BACK:  # Go back
                step = 2
                continue
            step = 4

        elif step == 4:
            # Final step - show results
            print("\n=== Results ===")
            print(f"Name: {name}")
            print(f"Age: {age}")
            print(f"Favorite Color: {color}")
            return


if __name__ == "__main__":
    simple_wizard_with_back()

#!/usr/bin/env python3

def println(*objects):
    """Print objects with empty line after."""
    print(*objects, end="\n\n")

def lnprintln(*objects):
    """Print objects with empty line before and after."""
    print()
    print(*objects, end="\n\n")
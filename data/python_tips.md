# Python Tips for Everyday Coding

## List Comprehensions

Instead of writing loops to build lists, use comprehensions — they're faster
and more readable.

    squares = [x * x for x in range(10)]
    evens = [x for x in numbers if x % 2 == 0]

## Enumerate Over Loops

When you need both the index and the value, don't use range(len(...)). Use
enumerate:

    for i, name in enumerate(names):
        print(i, name)

## f-Strings

The cleanest way to format strings in modern Python (3.6+):

    name = "Alice"
    print(f"Hello, {name}! You are {len(name)} letters long.")

## Context Managers

Always use `with` when dealing with files — it handles closing automatically
even if errors occur:

    with open("data.txt") as f:
        content = f.read()

## Virtual Environments

Never install packages globally. Always create a venv per project:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Useful Standard Library Gems

- `collections.Counter` — count hashable objects easily
- `itertools.chain` — flatten nested iterables
- `functools.lru_cache` — memoize pure functions
- `pathlib.Path` — modern, clean path handling (better than os.path)

## Debugging Tips

- `breakpoint()` drops you into pdb — no need to import anything.
- `print(repr(x))` shows the exact representation, not just str.
- Use type hints + mypy to catch bugs before runtime.

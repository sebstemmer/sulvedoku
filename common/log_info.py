def log_info(
        label: str,
        info: str | None = None,
        newline: bool = True
) -> None:
    if newline:
        print("\n")

    print(label)

    if info is not None:
        print(info)

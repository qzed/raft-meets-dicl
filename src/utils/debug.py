def run(function, *args, debug=True, **kwargs):
    if not debug:
        return function(*args, **kwargs)

    else:
        try:
            return function(*args, **kwargs)

        except Exception:
            import pdb
            import traceback

            traceback.print_exc()

            print()
            print('-- entering debugger '.ljust(80, '-'))
            print()

            pdb.post_mortem()

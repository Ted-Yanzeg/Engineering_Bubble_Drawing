# -*- coding: utf-8 -*-
from .cli import build_cli

def main():
    ap = build_cli()
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

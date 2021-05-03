#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mai 03, 10:37:26
@last modified : 2021 Mai 03, 10:42:21
"""

from colorama import Fore, Back, Style, init

init(autoreset=False)


class Printer:
    SECTION = -1
    SUBSECTION = -1

    @classmethod
    def print(cls, *args, **kwargs):
        print("> ", *args, **kwargs)

    @classmethod
    def section(cls, *args, **kwargs):
        Printer.SECTION += 1
        Printer.SUBSECTION = -1
        section_str = Fore.CYAN + f"[{cls.SECTION}] :: "
        print(section_str, *args, **kwargs)
        print(Style.RESET_ALL, end="\n")

    @classmethod
    def subsection(cls, *args, **kwargs):
        Printer.SUBSECTION += 1
        subsection_str = Fore.RED + f"({cls.SUBSECTION}) : "
        print(subsection_str, *args, **kwargs)
        print(Style.RESET_ALL, end="\n")


pprint = Printer.print
section = Printer.section
subsection = Printer.subsection

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os


class Arguments:
    def __init__(self, confFile):
        if not os.path.exists(confFile):
            raise Exception("The argument file does not exist: " + confFile)
        self.confFile = confFile

    def is_int(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_bool(self, s):
        return s.lower() == "true" or s.lower() == "false"

    # def readHyperDriveArguments(self, arguments):
    #     hyperdrive_opts = {}
    #     for i in range(0, len(arguments), 2):
    #         hp_name, hp_value = arguments[i:i+2]
    #         hp_name = hp_name.replace("--", "")
    #         if self.is_int(hp_value):
    #             hp_value = int(hp_value)
    #         elif self.is_float(hp_value):
    #             hp_value = float(hp_value)
    #         hyperdrive_opts[hp_name] = hp_value
    #     return hyperdrive_opts

    def add_opt(self, opt, key, value, force_override=False):
        if not key in opt or force_override:
            opt[key] = value
            if self.is_int(value):
                opt[key] = int(value)
            elif self.is_float(value):
                opt[key] = float(value)
            elif self.is_bool(value):
                opt[key] = value.lower() == "true"
        else:
            print("Warning: Option key %s already exists" % key)

    def readArguments(self):
        """
        Parse config file.

        Supported syntax:
         - general form: var WHITESPACE val, with WHITESPACE=space or TAB
         - whole-line or line-end comments begin with #
         - lines that end with backslash are continuation lines
         - multiple values are white-space separated, hence no spaces allowed in keys or values
        """
        opt = {}
        with open(self.confFile, encoding="utf-8") as f:
            prev_line = ""  # allow multi-line arguments
            for line in f:
                # concatenate previous line if it ended in backslash
                line = prev_line + line.strip()
                if line.endswith("\\"):
                    prev_line = line[:-1] + " "
                    continue
                prev_line = ""
                l = line.replace("\t", " ")
                # strip comments
                pos = l.find("#")
                if pos >= 0:
                    l = l[:pos]
                parts = l.split()
                if not parts:
                    continue  # empty line or line comment
                elif len(parts) == 1:
                    key = parts[0]
                    if not key in opt:
                        opt[key] = True
                else:
                    key = parts[0]
                    value = " ".join(parts[1:])
                    self.add_opt(opt, key, value)
            assert not prev_line, "Config file must not end with a backslash"
        return opt

class Colors(object):
    reset   = "\033[0m"
    red     = "\033[91m"
    green   = "\033[92m"
    yellow  = "\033[93m"
    blue    = "\033[94m"
    magenta = "\033[95m"
    cyan    = "\033[96m"
    white   = "\033[97m"

class Mode(object):
    INFO = "info"
    WARNING = "warning"
    IMPORTANCE = "importance"

class Debug(object):
    def __init__(self):
        self.color = Colors()
        self.mode = Mode()
        
    def shape(self, name, value, mode=Mode.INFO):
        if(mode == Mode.INFO):
            print(f"{self.color.green}{name}.shape={value.shape}{self.color.reset}")

        elif(mode == Mode.WARNING):
            print(f"{self.color.yellow}{name}.shape={value.shape}{self.color.reset}")

        elif(mode == Mode.IMPORTANCE):
            print(f"{self.color.red}{name}.shape={value.shape}{self.color.reset}")

        else:
            print(f"{self.color.red}The mode is not set correctly{self.color.reset}")

    def dtype(self, name, value, mode=Mode.INFO):
        if(mode == Mode.INFO):
            print(f"{self.color.green}{name}.type={value.dtype}{self.color.reset}")

        elif(mode == Mode.WARNING):
            print(f"{self.color.yellow}{name}.type={value.dtype}{self.color.reset}")

        elif(mode == Mode.IMPORTANCE):
            print(f"{self.color.red}{name}.type={value.dtype}{self.color.reset}")

        else:
            print(f"{self.color.red}The mode is not set correctly{self.color.reset}")

    def value(self, name, value, mode=Mode.INFO):
        if(mode == Mode.INFO):
            print(f"{self.color.green}{name}={value}{self.color.reset}")

        elif(mode == Mode.WARNING):
            print(f"{self.color.yellow}{name}={value}{self.color.reset}")

        elif(mode == Mode.IMPORTANCE):
            print(f"{self.color.red}{name}={value}{self.color.reset}")

        else:
            print(f"{self.color.red}The mode is not set correctly{self.color.reset}")

    def print(self, msg, mode=Mode.INFO):
        if(mode == Mode.INFO):
            print(f"{self.color.green}{msg}{self.color.reset}")

        elif(mode == Mode.WARNING):
            print(f"{self.color.yellow}{msg}{self.color.reset}")

        elif(mode == Mode.IMPORTANCE):
            print(f"{self.color.red}{msg}{self.color.reset}")

        else:
            print(f"{self.color.red}The mode is not set correctly{self.color.reset}")

